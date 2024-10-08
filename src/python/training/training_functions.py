""" Training functions for the different models. """
from collections import OrderedDict
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.losses.adversarial_loss import PatchAdversarialLoss
from diffusers.training_utils import compute_snr
from pynvml.smi import nvidia_smi
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from util import log_controlnet_samples, log_ldm_sample, log_reconstructions, log_image_pairs, sample_from_ldm, sample_from_controlnet, weighted_loss, compute_weight_tensor, clip_grad
from tqdm import tqdm
import random

REPORT = 'report'
REPORT_RAW = 'report_raw'
SOURCE = 'source'
TARGET = 'target'
PATIENT_ID = 'patient_id'
ACQ_TIME = 'acquisition_time'
SEGMENTATION_MASK = 'segmentation_mask'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        logging.info("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            logging.info(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")


# ----------------------------------------------------------------------------------------------------------------------
# AUTOENCODER KL
# ----------------------------------------------------------------------------------------------------------------------
def train_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    adv_weight: float,
    perceptual_weight: float,
    kl_weight: float,
    adv_start: int,
    early_stopping_after_num_epochs: int,
    eval_first: bool = False,
    clip_grad_norm_by: float = None,
    clip_grad_norm_or_value: float = 'norm',
    weight_loss_with_mask_by: int = None,
) -> float:
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    # setting the best loss epoch (used for early stopping to be the start epoch)
    best_loss_epoch = start_epoch

    raw_model = model.module if hasattr(model, "module") else model
    if best_loss > 999 or eval_first: # if inf
        logging.info(f"Now calculating val_loss of model {str(model.__class__.__name__)}.. ")
        val_loss = eval_aekl(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=val_loader,
            device=device,
            step=len(train_loader) * start_epoch,
            writer=writer_val,
            kl_weight=kl_weight,
            adv_weight=adv_weight if start_epoch >= adv_start else 0.0,
            perceptual_weight=perceptual_weight,
        )
    else:
        # this is to aviod another eval before continuing to train a checkpoint.
        val_loss = best_loss

    logging.info(f"epoch {start_epoch} val loss: {val_loss}")
    for epoch in tqdm(range(start_epoch, n_epochs)):
        train_epoch_aekl(
            model=model,
            discriminator=discriminator,
            perceptual_loss=perceptual_loss,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            writer=writer_train,
            kl_weight=kl_weight,
            adv_weight=adv_weight if epoch >= adv_start else 0.0,
            perceptual_weight=perceptual_weight,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
            clip_grad_norm_by=clip_grad_norm_by,
            clip_grad_norm_or_value=clip_grad_norm_or_value,
            weight_loss_with_mask_by=weight_loss_with_mask_by,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_aekl(
                model=model,
                discriminator=discriminator,
                perceptual_loss=perceptual_loss,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                kl_weight=kl_weight,
                adv_weight=adv_weight if epoch >= adv_start else 0.0,
                perceptual_weight=perceptual_weight,
            )
            logging.info(f"epoch {epoch + 1} val loss: {val_loss}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "best_loss": best_loss,
                "run_dir": run_dir,
            }
            if val_loss <= best_loss:
                vae_path = str(run_dir / f"vae_best_model.pth")
                logging.info(f"New best val loss {val_loss} after epoch {epoch + 1}. Saving model in {vae_path}")
                best_loss = val_loss
                torch.save(checkpoint, vae_path) # current best model (will be overwritten)
                torch.save(checkpoint, str(run_dir / f"vae_best_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth")) # model at that epoch (will be persistently stored).

            # Early stopping of AEKL
            if epoch + 1 > best_loss_epoch + early_stopping_after_num_epochs:
                logging.info(f"Val loss {best_loss} did not improve for {early_stopping_after_num_epochs} epochs (now: {epoch} best epoch: {best_loss_epoch}). Stopping training.")
                break

    logging.info(f"Training finished (epoch: {epoch + 1}, val_loss: {val_loss}).")
    # Save checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "best_loss": best_loss,
        "run_dir": run_dir,
    }
    final_model_path = str(run_dir / f"vae_final_model_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth")
    logging.info(f"Saving final model in {final_model_path}.")
    torch.save(checkpoint, final_model_path)
    return val_loss


def train_epoch_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
    clip_grad_norm_by: float = None,
    clip_grad_norm_or_value: float = 'norm',
    weight_loss_with_mask_by: int = None,
) -> None:
    model.train()
    discriminator.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        #logging.info(x)
        images = x[TARGET].to(device)

        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = model(x=images)
            # scaling the loss for segmented areas (e.g., the breast region)
            if weight_loss_with_mask_by is None:
                l1_loss = F.l1_loss(reconstruction.float(), images.float())
            else:
                segmentation_mask_batch = x[SEGMENTATION_MASK].to(device)
                weight_tensor = compute_weight_tensor(segmentation_mask_batch=segmentation_mask_batch, weight=weight_loss_with_mask_by)
                l1_loss = weighted_loss(loss_function='MAE', input_tensor=reconstruction.float(), target_tensor=images.float(), weight_tensor=weight_tensor)

            p_loss = perceptual_loss(reconstruction.float(), images.float())

            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
            )

        scaler_g.scale(losses["loss"]).backward()

        norm, norm_after = clip_grad(model=model, clip_grad_at=clip_grad_norm_by, norm_or_grad=clip_grad_norm_or_value, step=step, losses=losses, epoch=epoch)

        scaler_g.unscale_(optimizer_g)
        scaler_g.step(optimizer_g)
        scaler_g.update()

        # DISCRIMINATOR
        if adv_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                d_loss = adv_weight * discriminator_loss
                d_loss = d_loss.mean()

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            norm_D, norm_D_after = clip_grad(model = discriminator, clip_grad_at=1.0, norm_or_grad='norm', step=step,
                             losses=losses, epoch=epoch) # 'norm' instead of 'value' is adopted from previous implementation here
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            discriminator_loss = torch.tensor([0.0]).to(device)

        losses["d_loss"] = discriminator_loss

        #writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader) + step)
        #writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader) + step)

        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)
        train_update_dict = {
                "epoch": epoch,
                "norm_D_before_clip_at1.0": f"{norm_D:.6f}",
                "norm_D_after_clip_at1.0": f"{norm_D_after:.6f}",
                f"norm_AEKL_before_clip_at{clip_grad_norm_by}": norm,
                f"norm_AEKL_after_clip_at{clip_grad_norm_by}": norm_after,
                "loss": f"{losses['loss'].item():.6f}",
                "l1_loss": f"{losses['l1_loss'].item():.6f}",
                "p_loss": f"{losses['p_loss'].item():.6f}",
                "g_loss": f"{losses['g_loss'].item():.6f}",
                "d_loss": f"{losses['d_loss'].item():.6f}",
                "lr_g": f"{get_lr(optimizer_g):.6f}",
                "lr_d": f"{get_lr(optimizer_d):.6f}",
            }
        logging.info(train_update_dict)
        pbar.set_postfix(train_update_dict,)


@torch.no_grad()
def eval_aekl(
    model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    kl_weight: float,
    adv_weight: float,
    perceptual_weight: float,
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    total_losses = OrderedDict()
    for x in tqdm(loader):
        #logging.info(x)
        images = x[TARGET].to(device)

        with autocast(enabled=True):
            # GENERATOR
            reconstruction, z_mu, z_sigma = model(x=images)

            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            else:
                generator_loss = torch.tensor([0.0]).to(device)

            # DISCRIMINATOR
            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            else:
                discriminator_loss = torch.tensor([0.0]).to(device)

            loss = l1_loss + kl_weight * kl_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            kl_loss = kl_loss.mean()
            g_loss = generator_loss.mean()
            d_loss = discriminator_loss.mean()

            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                kl_loss=kl_loss,
                g_loss=g_loss,
                d_loss=d_loss,
            )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    #for k, v in total_losses.items():
    #    writer.add_scalar(f"{k}", v, step)
    #    writer.add_scalar(f"{k}", v, step)

    logging.info(f"total losses in eval after step {step}: {total_losses}")
    log_reconstructions(
        image=images,
        reconstruction=reconstruction,
        writer=writer,
        step=step,
    )

    return total_losses["l1_loss"]


# ----------------------------------------------------------------------------------------------------------------------
# Latent Diffusion Model
# ----------------------------------------------------------------------------------------------------------------------
def train_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    scale_factor: float = 1.0,
    eval_first: bool = False,
    cond_on_acq_times: bool = False,
    clip_grad_norm_by: float = None,
    clip_grad_norm_or_value: float = 'norm',
    snr_gamma: float = None, # 5 is recommended https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L351
    lr_scheduler=None, # https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L972C17-L972C36
) -> float:
    scaler = GradScaler()
    raw_model = model.module if hasattr(model, "module") else model
    if eval_first:
        logging.info(f"Now calculating val_loss of model {str(model.__class__.__name__)}. Best loss before was {best_loss}. ")
        val_loss = eval_ldm(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            text_encoder=text_encoder,
            loader=val_loader,
            device=device,
            step=len(train_loader) * start_epoch,
            writer=writer_val,
            sample=True,
            scale_factor=scale_factor,
            break_at_iteration=2,
            cond_on_acq_times=cond_on_acq_times,
        )
    else:
        val_loss = best_loss

    logging.info(f"epoch {start_epoch} val loss: {val_loss}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_ldm(
            model=model,
            stage1=stage1,
            scheduler=scheduler,
            text_encoder=text_encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
            cond_on_acq_times=cond_on_acq_times,
            clip_grad_norm_by = clip_grad_norm_by,
            clip_grad_norm_or_value=clip_grad_norm_or_value,
            snr_gamma=snr_gamma,
            lr_scheduler=lr_scheduler,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_ldm(
                model=model,
                stage1=stage1,
                scheduler=scheduler,
                text_encoder=text_encoder,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                sample=True, # if (epoch + 1) % (eval_freq * 2) == 0 else False,
                scale_factor=scale_factor,
                cond_on_acq_times=cond_on_acq_times,
            )

            logging.info(f"epoch {epoch + 1} val loss: {val_loss}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "diffusion": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }
            #checkpoint_path = str(run_dir / "diffusion_checkpoint.pth")
            #torch.save(checkpoint, checkpoint_path)

            if val_loss <= best_loss:
                model_path = str(run_dir / "diffusion_best_model.pth")
                logging.info(f"New best val loss {val_loss} after epoch {epoch + 1}. Saving model in {model_path}")
                best_loss = val_loss
                torch.save(checkpoint, model_path) # current best model (will be overwritten)
                torch.save(checkpoint, str(run_dir / f"ldm_best_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth")) # model at that epoch (will be persistently stored).
            elif (epoch +1) % 20 == 0:
                # we also want a routine saving of the model every 20 epochs
                torch.save(checkpoint, str(run_dir / f"ldm_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth"))

    logging.info(f"Training finished (epoch: {epoch + 1}).")
    checkpoint = {
        "epoch": epoch + 1,
        "diffusion": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    final_model_path = str(run_dir / f"ldm_final_model_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth")
    logging.info(f"Saving final model in {final_model_path}.")
    torch.save(checkpoint, final_model_path)
    return val_loss


def train_epoch_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
    scale_factor: float = 1.0,
    cond_on_acq_times: bool = False,
    clip_grad_norm_by:float = None,
    clip_grad_norm_or_value: float = 'norm',
    snr_gamma: float = None, # 5 is recommended https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L351
    lr_scheduler=None, # https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L972C17-L972C36
) -> None:
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x[TARGET].to(device)
        reports = x[REPORT].to(device)
        reports_raw = x[REPORT_RAW] # not a tensor
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
        # second passed since precontrast image acquisition
        acquisition_time = x[ACQ_TIME].to(device) if cond_on_acq_times else None
        if acquisition_time is not None:
            acquisition_time = acquisition_time.unsqueeze(1)
            logging.debug(f"acquisition_time shape: {acquisition_time.shape}")

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                e = stage1(images) * scale_factor

            prompt_embeds = text_encoder(reports.squeeze(1))
            prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
            noise_pred = sample_from_ldm(model=model,
                                         noisy_e=noisy_e,
                                         timesteps=timesteps,
                                         prompt_embeds=prompt_embeds,
                                         class_labels=acquisition_time if cond_on_acq_times else None
            )

            # TODO: Consider adding the guidance_scale to the noise_pred during training:
            #  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1013-L1015
            #  https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#denoise-the-image

            # TODO: Consider noise rescaling, as in:
            #  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1017-L1019

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise

        if snr_gamma is None:
            loss = F.mse_loss(noise_pred.float(), target.float())
        elif scheduler.alphas_cumprod is None:
            # If the scheduler is from monai-generative, we may not be able to use diffusers compute_snr function
            # scheduler needs to have self.alphas_cumprod defined.
            # See https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/src/diffusers/training_utils.py#L43
            logging.error(f"scheduler.alphas_cumprod of {scheduler.__class__.__name__} is None. Please use a scheduler from diffusers if you want to use snr_gamma={snr_gamma}.")
        else:
            # Using codeblock from https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L944-L961
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if scheduler.prediction_type == 'epsilon' or (hasattr(scheduler, "config") and scheduler.config.prediction_type == "epsilon"):
                mse_loss_weights = mse_loss_weights / snr
            elif scheduler.prediction_type == "v_prediction" or (hasattr(scheduler, "config") and scheduler.config.prediction_type == "v_prediction"):
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()

        # gradients need to be unscaled before gradient clipping is applied.
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        scaler.unscale_(optimizer) # Unscales the gradients of optimizer's assigned params in-place

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        norm, norm_after = clip_grad(model=model, clip_grad_at=clip_grad_norm_by, norm_or_grad=clip_grad_norm_or_value, step=step, losses=losses, epoch=epoch)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
        #writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        if lr_scheduler is not None:
            lr_scheduler.step()

        #writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        #for k, v in losses.items():
        #    writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

        epoch_information_dict = {"epoch": epoch, f"norm_before_clip_at{clip_grad_norm_by}": norm, f"norm_after_clip": norm_after, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"}
        logging.debug(f"epoch info in ldm training after step {step}: {epoch_information_dict}")
        pbar.set_postfix(epoch_information_dict)


@torch.no_grad()
def eval_ldm(
    model: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    sample: bool = True,
    scale_factor: float = 1.0,
    break_at_iteration: int = -1,
    cond_on_acq_times: bool = False,
) -> float:
    model.eval()
    stage1.eval()
    raw_stage1 = stage1.module if hasattr(stage1, "module") else stage1
    raw_model = model.module if hasattr(model, "module") else model
    total_losses = OrderedDict()
    
    counter = 0
    for x in tqdm(loader):
        images = x[TARGET].to(device)
        reports = x[REPORT].to(device)
        reports_raw = x[REPORT_RAW] # not a tensor
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
        # second passed since precontrast image acquisition
        acquisition_time = x[ACQ_TIME].to(device) if cond_on_acq_times else None
        if acquisition_time is not None:
            acquisition_time = acquisition_time.unsqueeze(1)
            logging.debug(f"acquisition_time shape: {acquisition_time.shape}")

        if sample and counter == -1:
            # first, let's have a look at the output of the dataloader
            # if isinstance(x[SOURCE], str):
            log_image_pairs(image=images, reconstruction=images, model_type='ldm', writer=writer, step=step,
                            title='ldm_test_image_target')
            # else:

        with autocast(enabled=True):
            e_pre_scaling = stage1(images)
            e = e_pre_scaling * scale_factor

            prompt_embeds = text_encoder(reports.squeeze(1))
            prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)
            logging.debug(f"class_labels (acqusition times): {acquisition_time} cond_on_acq_times: {cond_on_acq_times}")
            noise_pred = sample_from_ldm(model=model,
                                         noisy_e=noisy_e,
                                         timesteps=timesteps,
                                         prompt_embeds=prompt_embeds,
                                         class_labels=acquisition_time
            )

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                # TODO: Change target noise calculation if scheduler is based on diffusers library (e.g. PNDMScheduler) (get_velocity function not in diffusers)
                # https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/schedulers/scheduler.py#L191
                # https://github.com/huggingface/diffusers/blob/v0.3.0/src/diffusers/schedulers/scheduling_ddpm.py#L248
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            loss = F.mse_loss(noise_pred.float(), target.float())

        if break_at_iteration > 0:
            logging.info(
                f"noisy_e shape: {noisy_e.shape}, e shape: {e.shape}, prompt_embeds shape: {prompt_embeds.shape}, timesteps shape: {timesteps.shape},"
                f"noise shape: {noise.shape}, noise_pred shape: {noise_pred.shape}, target shape: {target.shape}")

            # Let's print some statistics of latent of AEKL and LDM noise vector z.
            logging.info(
                f"\nstep: {step} \n"
                f"scale factor: {scale_factor} \n"
                f"{stage1.__class__.__name__} z mean (std): {torch.round(torch.mean(e_pre_scaling), decimals=3)} ({torch.round(torch.std(e_pre_scaling), decimals=3)}) \n"
                f"{stage1.__class__.__name__} z mean (std) after scaling: {torch.round(torch.mean(e), decimals=3)} ({torch.round(torch.std(e), decimals=3)}) \n"
                f"torch random noise mean (std): {torch.round(torch.mean(noise), decimals=3)} ({torch.round(torch.std(noise), decimals=3)}) \n"
                f"torch random noise mean (std) after scheduler add_noise: {torch.round(torch.mean(noisy_e), decimals=3)} ({torch.round(torch.std(noisy_e), decimals=3)}) \n"
                f"{model.__class__.__name__}'s predicted noise mean (std): {torch.round(torch.mean(noise_pred), decimals=3)} ({torch.round(torch.std(noise_pred), decimals=3)}) \n"
                f"{scheduler.prediction_type}-based target noise mean (std): {torch.round(torch.mean(target), decimals=3)} ({torch.round(torch.std(target), decimals=3)})\n")

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]
        if break_at_iteration > 0 and counter == break_at_iteration:
            break
        counter = counter +1
    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    #for k, v in total_losses.items():
    #    writer.add_scalar(f"{k}", v, step)

    logging.info(f"total losses in ldm eval after step {step}: {total_losses}")

    if sample:

        # conditional
        log_ldm_sample(
            model=raw_model,
            stage1=raw_stage1,
            scheduler=scheduler,
            text_encoder=text_encoder,
            spatial_shape=tuple(e.shape[1:]),
            writer=writer,
            step=step,
            device=device,
            scale_factor=scale_factor,
            text_condition=reports[0],
            text_condition_raw=reports_raw[0],
            class_labels=acquisition_time[0] if cond_on_acq_times else None,
        )
        # unconditional
        log_ldm_sample(
            model=raw_model,
            stage1=raw_stage1,
            scheduler=scheduler,
            text_encoder=text_encoder,
            spatial_shape=tuple(e.shape[1:]),
            writer=writer,
            step=step,
            device=device,
            scale_factor=scale_factor,
            class_labels=acquisition_time[0] if cond_on_acq_times else None,
        )

    return total_losses["loss"]


# ----------------------------------------------------------------------------------------------------------------------
# Controlnet
# ----------------------------------------------------------------------------------------------------------------------
def train_controlnet(
    controlnet: nn.Module,
    diffusion: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    start_epoch: int,
    best_loss: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    eval_freq: int,
    writer_train: SummaryWriter,
    writer_val: SummaryWriter,
    device: torch.device,
    run_dir: Path,
    scale_factor: float = 1.0,
    eval_first: bool = False,
    cond_on_acq_times: bool = False,
    clip_grad_norm_by: float = None,
    clip_grad_norm_or_value: float = 'norm',
    controlnet_conditioning_scale: float = 1.0,
    snr_gamma: float = None, # 5 is recommended https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L351
    lr_scheduler=None, # https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L972C17-L972C36
) -> float:
    scaler = GradScaler()
    raw_controlnet = controlnet.module if hasattr(controlnet, "module") else controlnet

    if eval_first:
        logging.info(
            f"Now calculating val_loss of model {str(controlnet.__class__.__name__)} (based on "
            f"ldm={str(diffusion.__class__.__name__)} and "
            f"AE={str(stage1.__class__.__bases__[0].__name__)}). "
            f"Best loss before was {best_loss}. ")
        val_loss = eval_controlnet(
            controlnet=controlnet,
            diffusion=diffusion,
            stage1=stage1,
            scheduler=scheduler,
            text_encoder=text_encoder,
            loader=val_loader,
            device=device,
            step=len(train_loader) * start_epoch,
            writer=writer_val,
            scale_factor=scale_factor,
            break_at_iteration=2,
            cond_on_acq_times=cond_on_acq_times,
            sample=True,  # if (epoch + 1) % (eval_freq * 2) == 0 else False,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
    else:
        val_loss = best_loss

    logging.info(f"epoch {start_epoch} val loss: {val_loss}")

    for epoch in range(start_epoch, n_epochs):
        train_epoch_controlnet(
            controlnet=controlnet,
            diffusion=diffusion,
            stage1=stage1,
            scheduler=scheduler,
            text_encoder=text_encoder,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer_train,
            scaler=scaler,
            scale_factor=scale_factor,
            cond_on_acq_times=cond_on_acq_times,
            clip_grad_norm_by=clip_grad_norm_by,
            clip_grad_norm_or_value=clip_grad_norm_or_value,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            snr_gamma=snr_gamma,
            lr_scheduler=lr_scheduler,
        )

        if (epoch + 1) % eval_freq == 0:
            val_loss = eval_controlnet(
                controlnet=controlnet,
                diffusion=diffusion,
                stage1=stage1,
                scheduler=scheduler,
                text_encoder=text_encoder,
                loader=val_loader,
                device=device,
                step=len(train_loader) * epoch,
                writer=writer_val,
                scale_factor=scale_factor,
                cond_on_acq_times=cond_on_acq_times,
                sample=True,  # if (epoch + 1) % (eval_freq * 2) == 0 else False,
            )

            logging.info(f"epoch {epoch + 1} val loss: {val_loss}")
            print_gpu_memory_report()

            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "controlnet": controlnet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
            }

            if val_loss <= best_loss:
                controlnet_path = str(run_dir / "controlnet_best_model.pth")
                logging.info(f"New best val loss {val_loss} after epoch {epoch + 1}. Saving model in {controlnet_path}")
                best_loss = val_loss
                torch.save(checkpoint, controlnet_path) # current best model (will be overwritten)
                torch.save(checkpoint, str(run_dir / f"controlnet_best_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth")) # model at that epoch (will be persistently stored).
            elif (epoch +1) % 20 == 0:
                # we also want a routine saving of the model every 20 epochs
                torch.save(checkpoint, str(run_dir / f"controlnet_best_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth")) # model at that epoch (will be persistently stored).

    logging.info(f"Training finished (epoch: {epoch + 1}, val_loss: {val_loss}).")
    checkpoint = {
        "epoch": epoch + 1,
        "controlnet": controlnet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    final_model_path = str(run_dir / f"controlnet_final_model_val_loss_{str(val_loss).replace('.','').replace(',','')}_epoch{epoch + 1}.pth")
    controlnet_final_path = str(run_dir / "controlnet_final_model.pth")
    logging.info(f"Saving final model in {final_model_path}.")
    torch.save(checkpoint, controlnet_final_path)

    return val_loss


def train_epoch_controlnet(
    controlnet: nn.Module,
    diffusion: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler,
    scale_factor: float = 1.0,
    cond_on_acq_times: bool = False,
    clip_grad_norm_by: float=None,
    clip_grad_norm_or_value: float = 'norm',
    controlnet_conditioning_scale: float = 1.0,
    snr_gamma: float = None, # 5 is recommended https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L351
    lr_scheduler = None, # https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L972C17-L972C36
) -> None:
    controlnet.train()

    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x[TARGET].to(device)
        reports = x[REPORT].to(device)
        reports_raw = x[REPORT_RAW] # not a tensor
        cond = x[SOURCE].to(device)
        acquisition_time = x[ACQ_TIME].to(device) if cond_on_acq_times else None
        if acquisition_time is not None:
            acquisition_time = acquisition_time.unsqueeze(0)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                e = stage1(images) * scale_factor

            prompt_embeds = text_encoder(reports.squeeze(1))
            prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            down_block_res_samples, mid_block_res_sample = sample_from_controlnet(
                controlnet=controlnet,
                noisy_e=noisy_e,
                controlnet_cond=cond,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                class_labels=acquisition_time if cond_on_acq_times else None,
                conditioning_scale=controlnet_conditioning_scale,
            )

            noise_pred = sample_from_ldm(model=diffusion,
                                               noisy_e=noisy_e,
                                               timesteps=timesteps,
                                               prompt_embeds=prompt_embeds,
                                               down_block_res_samples=down_block_res_samples,
                                               mid_block_res_sample = mid_block_res_sample,
                                               class_labels=acquisition_time if cond_on_acq_times else None,
                                         )
            # TODO: Consider adding the guidance_scale to the noise_pred during training:
            #  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1013-L1015
            #  https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#denoise-the-image

            # TODO: Consider noise rescaling, as in:
            #  https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1017-L1019

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            random_sampling = random.randint(0, 9999) == 1 # randomly sample every 10000 steps on average. More samples trade off training speed
            if step % 12000 == 0 or random_sampling:
                # we randomly log a training sample to qualitatively see if the model is converging well.
                # preparing acq time
                if cond_on_acq_times:
                    acquisition_time = torch.squeeze(acquisition_time, 0)
                    acquisition_time = torch.unsqueeze(acquisition_time, 1)[0]
                log_controlnet_samples(
                    cond=cond[0],
                    target=images[0],
                    controlnet=controlnet,
                    diffusion=diffusion,
                    text_encoder=text_encoder,
                    spatial_shape=tuple(e.shape[1:]),
                    scheduler=scheduler,
                    stage1=stage1,
                    writer=writer,
                    step=step,
                    device=device,
                    scale_factor=scale_factor,
                    text_condition=reports[0],
                    text_condition_raw=reports_raw[0],
                    acquisition_time=acquisition_time if cond_on_acq_times else None,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    dataset_name="training",
                    epoch=epoch,
                )

        if snr_gamma is None:
            loss = F.mse_loss(noise_pred.float(), target.float())
        elif scheduler.alphas_cumprod is None:
            # If the scheduler is from monai-generative, we may not be able to use diffusers compute_snr function
            # scheduler needs to have self.alphas_cumprod defined.
            # See https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/src/diffusers/training_utils.py#L43
            logging.error(f"scheduler.alphas_cumprod of {scheduler.__class__.__name__} is None. Please use a scheduler from diffusers if you want to use snr_gamma={snr_gamma}.")
        else:
            # Using codeblock from https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L944-L961
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            if scheduler.prediction_type == 'epsilon' or (hasattr(scheduler, "config") and scheduler.config.prediction_type == "epsilon"):
                mse_loss_weights = mse_loss_weights / snr
            elif scheduler.prediction_type == "v_prediction" or (hasattr(scheduler, "config") and scheduler.config.prediction_type == "v_prediction"):
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        losses = OrderedDict(loss=loss)

        scaler.scale(losses["loss"]).backward()

        # gradients need to be unscaled before gradient clipping is applied.
        # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        scaler.unscale_(optimizer) # Unscales the gradients of optimizer's assigned params in-place

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        norm, norm_after = clip_grad(model=controlnet, clip_grad_at=clip_grad_norm_by, norm_or_grad=clip_grad_norm_or_value, step=step, losses=losses, epoch=epoch)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
        #writer.add_scalar("lr", get_lr(optimizer), epoch * len(loader) + step)

        # Update the lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        #for k, v in losses.items():
        #   writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)
        epoch_information_dict = {"epoch": epoch, f"norm_before_clip_at{clip_grad_norm_by}": norm, f"norm_after_clip": norm_after, "loss": f"{losses['loss'].item():.5f}", "lr": f"{get_lr(optimizer):.6f}"}
        logging.debug(f"epoch info in {controlnet.__class__.__name__} training after step {step}: {epoch_information_dict}")
        pbar.set_postfix(epoch_information_dict)


@torch.no_grad()
def eval_controlnet(
    controlnet: nn.Module,
    diffusion: nn.Module,
    stage1: nn.Module,
    scheduler: nn.Module,
    text_encoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    scale_factor: float = 1.0,
    break_at_iteration: int = -1,
    cond_on_acq_times: bool = False,
    sample: bool = True,
    controlnet_conditioning_scale: float = 1.0,
) -> float:
    controlnet.eval() # diffusion and stage1 alread in eval() mode
    total_losses = OrderedDict()
    counter = 0

    for x in tqdm(loader):
        images = x[TARGET].to(device)
        reports = x[REPORT].to(device)
        reports_raw = x[REPORT_RAW] # not a tensor
        cond = x[SOURCE].to(device)
        acquisition_time = x[ACQ_TIME].to(device) if cond_on_acq_times else None
        if acquisition_time is not None:
            acquisition_time = acquisition_time.unsqueeze(0)

        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        if sample and counter == 0:
            # first, let's have a look at the output of the dataloader
            # if isinstance(x[SOURCE], str):
            log_image_pairs(image=cond, reconstruction=images, model_type='controlnet', writer=writer, step=step,
                            title='controlnet_input_image_pairs')

        with autocast(enabled=True):
            e = stage1(images) * scale_factor

            prompt_embeds = text_encoder(reports.squeeze(1))
            prompt_embeds = prompt_embeds[0]

            noise = torch.randn_like(e).to(device)
            noisy_e = scheduler.add_noise(original_samples=e, noise=noise, timesteps=timesteps)

            if break_at_iteration >0:
                logging.info(f"noisy_e shape: {noisy_e.shape}, e shape: {e.shape}, prompt_embeds shape: {prompt_embeds.shape}, timesteps shape: {timesteps.shape}")

            down_block_res_samples, mid_block_res_sample = sample_from_controlnet(
                controlnet=controlnet,
                noisy_e=noisy_e,
                controlnet_cond=cond,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                class_labels=acquisition_time if cond_on_acq_times else None,
                conditioning_scale=controlnet_conditioning_scale,
            )
            # down_block_res_samples = torch.stack(down_block_res_samples).to(device) # TODO: why is stacking needed here?
            #mid_block_res_sample = torch.stack(mid_block_res_sample).to(device) # TODO: why is stacking needed here?

            # See input into unet2d_cond of ldm here: https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/unet_2d_condition.py#L845
            down_block_res_samples = tuple(down_block_res_samples)

            logging.debug(f"controlnet output: down_block_res_sample tensors with shapes = {[tensor_.shape for tensor_ in down_block_res_samples]}, shape mid_block_res_sample tensor = {(mid_block_res_sample.shape)}.")
            logging.debug(f"diffusion model input shapes: noisy_e.shape = {noisy_e.shape}, timesteps.shape = {timesteps.shape}, prompt_embeds.shape = {prompt_embeds.shape}")

            #logging.info(f"mid_block_res_sample shape: {mid_block_res_sample.shape}")
            #mid_block_res_sample = None

            noise_pred = sample_from_ldm(model=diffusion,
                                               noisy_e=noisy_e,
                                               timesteps=timesteps,
                                               prompt_embeds=prompt_embeds,
                                               down_block_res_samples=down_block_res_samples,
                                               mid_block_res_sample = mid_block_res_sample,
                                               class_labels=acquisition_time if cond_on_acq_times else None,
            )

            if scheduler.prediction_type == "v_prediction":
                # Use v-prediction parameterization
                target = scheduler.get_velocity(e, noise, timesteps)
            elif scheduler.prediction_type == "epsilon":
                target = noise
            if break_at_iteration >0:
                logging.info(f"noise shape: {noise.shape}, noise_pred shape: {noise_pred.shape}, target shape: {target.shape}")
            loss = F.mse_loss(noise_pred.float(), target.float())

        loss = loss.mean()
        losses = OrderedDict(loss=loss)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v.item() * images.shape[0]

        if break_at_iteration > 0 and counter == break_at_iteration:
            break
        counter = counter +1

    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    #for k, v in total_losses.items():
    #    writer.add_scalar(f"{k}", v, step)

    if sample:
        # conditional
        # prepare acq times
        if cond_on_acq_times:
            acquisition_time = torch.squeeze(acquisition_time, 0)
            acquisition_time = torch.unsqueeze(acquisition_time, 1)[0]
        log_controlnet_samples(
            cond=cond[0],
            target=images[0],
            controlnet=controlnet,
            diffusion=diffusion,
            text_encoder=text_encoder,
            spatial_shape=tuple(e.shape[1:]),
            scheduler=scheduler,
            stage1=stage1,
            writer=writer,
            step=step,
            device=device,
            scale_factor=scale_factor,
            text_condition=reports[0],
            text_condition_raw=reports_raw[0],
            acquisition_time=acquisition_time if cond_on_acq_times else None,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            dataset_name="validation"
        )
    return total_losses["loss"]
