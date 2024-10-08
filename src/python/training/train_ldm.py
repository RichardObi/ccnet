""" Training script for the diffusion model in the latent space of the previously trained AEKL model. """
import argparse
import warnings
from pathlib import Path
import logging

import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_ldm
from transformers import CLIPTextModel
from util import get_dataloader, setup_logger, adjust_class_embedding
from diffusers.optimization import get_scheduler
try:
    from diffusers import AutoencoderKL
except: # exception depends on diffusers library version
    from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers import UNet2DConditionModel # pretrained model

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--stage1_uri", help="Path readable by load_model.")
    parser.add_argument("--scale_factor", type=float, default=None, help="Path readable by load_model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")
    parser.add_argument("--is_resumed", action="store_true" , help="resume training from checkpoint in run_dir folder, if available..")
    parser.add_argument("--torch_detect_anomaly", type=int, default=0 , help="Detects NaN/Infs in the backward pass. Can slow down training significantly!")
    parser.add_argument("--use_pretrained", type=int, default=0, help="use a pretrained stage1 autoencoder instead of the trained checkpoint. 0 = False, 1 = True")
    parser.add_argument("--fine_tune", action="store_true" , help="Fine tune the LDM model. If false, the LDM model is trained from scratch.")
    parser.add_argument("--is_stage1_fine_tuned", action="store_true" , help="Info if the stage1 model was fine tuned, therefore requiring different ldm input and output dims.")
    parser.add_argument("--source_model", type=str, default="stabilityai/stable-diffusion-2-1-base", help="source model for the stage1 autoencoder and text_encoder") #" \
    parser.add_argument("--cond_on_acq_times", action="store_true" , help="If true, MRI acquisition times will be passed as conditional into ldm during train and eval .") #" \
    parser.add_argument("--clip_grad_norm_by", type=float,  default=None, help="Clip the gradient norm by this floating point value (default in torch is =2). SD training has produced NaNs when grad_norm > 2")
    parser.add_argument("--clip_grad_norm_or_value", type=str,  default='norm', help="Clip either the norm of the gradients or the value of the gradients. Norm keeps same direction while value can change direction. Default is 'norm'.")
    parser.add_argument("--img_width", type=int,  default=None, help="The image width that the dataloader will resize the input images to")
    parser.add_argument("--img_height", type=int,  default=None, help="The image height that the dataloader will resize the input images to")
    parser.add_argument("--checkpoint_name", type=str,  default="diffusion_best_model.pth", help="The checkpoint file name and extension.")
    parser.add_argument("--use_default_report_text", action="store_true" , help="If true, the default report text will be used for all samples returned from the dataloader. Otherwise, a custom report text will be loaded from the from the dataloader.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="The gamma value for the signal-to-noise ratio (SNR) calculation. Recommendation is 5.0")  # 5 is recommended in https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L355
    parser.add_argument("--lr_warmup_steps", type=int, default=None, help="Number of steps for the warmup in the lr scheduler. Recommendation is 500. ")  # 500 is recommended in https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L349
    #"stabilityai/stable-diffusion-xl-base-1.0"
    # Not implemented for now:
    #parser.add_argument("--early_stopping_after_num_epochs", type=int, default=99999, help="Stop training after this number of epochs passed without val loss improvement.")

    args = parser.parse_args()
    return args


class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for the DataParallel usage in the training loop."""

    def __init__(self, model: nn.Module, stack_input_to_rgb:bool =False) -> None:
        super().__init__()
        self.model = model
        self.stack_input_to_rgb = stack_input_to_rgb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1 and self.stack_input_to_rgb:
            x = torch.cat((x, x, x), dim=1)
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoder_kl.py#L443-L447
            posterior = self.model.encode(x).latent_dist
            z = posterior.sample() #mode()
            #dec = self.decode(z).sample
            #return (dec,)
        else:
            z_mu, z_sigma = self.model.encode(x)
            z = self.model.sampling(z_mu, z_sigma)
        return z


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    if args.torch_detect_anomaly > 0:
        # anomaly detection from torch # Watchout: This can slow down training significantly!
        torch.autograd.set_detect_anomaly(True)
        logging.info(f"Now detecting NaN/Infs in the backward pass with torch.autograd.set_detect_anomaly(True). This can slow down training significantly!")

    #output_dir = Path("/project/outputs/runs/")
    output_dir = Path("project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.run_dir
    checkpoint_path = None
    if Path(run_dir).exists() and Path(run_dir / args.checkpoint_name).exists():
        resume = True
        checkpoint_path = str(run_dir / args.checkpoint_name)
    elif Path(args.checkpoint_name).exists():
        resume = True
        checkpoint_path = Path(args.checkpoint_name)
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    setup_logger(run_dir / f"train_diffusion_{args.experiment}.log")

    logging.info(f"Run directory: {str(run_dir)}")
    logging.info(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    logging.info("Getting data...")
    cache_name = "cached_data_ldm_p1" if "P1" in str(run_dir) else "cached_data_ldm"
    cache_dir = output_dir / cache_name
    cache_dir.mkdir(exist_ok=True)
    device = torch.device("cuda")

    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        img_width=args.img_width,
        img_height=args.img_height,
        model_type="diffusion",
        use_default_report_text=args.use_default_report_text,
    )
    # Loading the config
    config = OmegaConf.load(args.config_file)

    # Load Autoencoder to produce the latent representations
    if args.use_pretrained>0:
        # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
        logging.info(f"use_pretrained={args.use_pretrained}: Loading pretrained stage 1 model from {args.source_model} vae")
        stage1 = AutoencoderKL.from_pretrained(args.source_model, subfolder="vae") # model needs 3-channel rgb inputs
        stage1 = Stage1Wrapper(model=stage1, stack_input_to_rgb=True)
        logging.info(f"Adjusting in_channels of ldm config from {config['ldm']['params']['in_channels']} to 4 to make it compatible with PRETRAINED stable diffusion (2-1, xl) vae outputs.")
        # 4 in and out embedding channels are standard in stability-ai's 2-1 and xl ldm models.
        config["ldm"]["params"]["in_channels"] = 4
        config["ldm"]["params"]["out_channels"] = 4
    else:
        if args.is_stage1_fine_tuned:
            logging.info(
                f"Adjusting in_channels of ldm config from {config['ldm']['params']['in_channels']} to 4 to make it compatible with FINE-TUNED stable diffusion (2-1, xl) vae outputs.")
            # 4 in and out embedding channels are standard in stability-ai's 2-1 and xl ldm models.
            config["ldm"]["params"]["in_channels"] = 4
            config["ldm"]["params"]["out_channels"] = 4
        logging.info(f"Loading stage 1 model from {args.stage1_uri}")
        if ".pt" in args.stage1_uri:
            logging.info(f"args.stage1_uri ({args.stage1_uri}) points to a ckpt file. Loading ckpt now..")
            stage1 = AutoencoderKL(**config["stage1"].get("params", dict()))
            try:
                ckpt = torch.load(args.stage1_uri, map_location=torch.device(device))
                stage1.load_state_dict(ckpt["vae"])
            except:
                stage1.load_state_dict(args.stage1_uri)
        else:
            stage1 = mlflow.pytorch.load_model(args.stage1_uri)
        stage1 = Stage1Wrapper(model=stage1)
    stage1.eval()

    if args.fine_tune:
        #config["ldm"]["base_lr"] = 2.5e-6 # fine-tune lr could be lower than the base_lr (e.g. 2.5e-5) used for training from scratch
        logging.info(f"fine_tune={args.fine_tune}: Creating LDM model to fine-tune based on {args.source_model} model with new fine-tune lr={config['ldm']['base_lr']}.")
        # Choice of diffusion and scheduler is based on https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/model_index.json
        # unet: https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/unet/config.json
        # Note, if args.source_model is changed, the diffusion and scheduler classed might need to be changed as well.
        diffusion = UNet2DConditionModel.from_pretrained(args.source_model, subfolder="unet")

        # TODO Go back to experiment with pndm scheduler whenever possible. Currently pndm scheduler is not working with the current version of diffusers.
        scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
        # We load the scheduler for the pretrained ldm with which it was trained. This scheduler might differ from the one used during manual training of the ldm.
        #scheduler = PNDMScheduler.from_pretrained(args.source_model, subfolder="scheduler") # https://arxiv.org/abs/2202.09778
        #if scheduler.prk_timesteps is None:
            # To avoid bug when invoking len on NoneType scheduler.prk_timesteps in line 256 of diffusers/schedulers/scheduling_pndm.py,
            # we reinitialize the PNDM scheduler based on the values from stable diffusion model 2-1
            # https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/scheduler/scheduler_config.json
            # prk steps are documented to be not useful for stable diffusion and are skipped are used in 2-1-base
            # https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/schedulers/scheduling_pndm.py#L204-L206

            # pndm_dict= dict(
            #     beta_end=0.012,
            #     beta_schedule="scaled_linear",
            #     beta_start=0.00085,
            #     #clip_sample=False,
            #     num_train_timesteps=1000,
            #     prediction_type="epsilon",
            #     set_alpha_to_one=False,
            #     skip_prk_steps=True,
            #     steps_offset=1,
            #     trained_betas=None,
            # )
            # logging.info(f"{scheduler.__class__.__name__} initialization from {args.source_model} resulted in error as"
            #              f" scheduler.prk_timesteps: {scheduler.prk_timesteps}. As fallback: we now manually initialize"
            #              f" {PNDMScheduler.__class__.__name__} with default values stability/stable-diffusion-2-1-base "
            #              f"values instead: {pndm_dict}. Please change code if this is not desired.")
            # scheduler = PNDMScheduler(**pndm_dict)

    else:
        # Create the diffusion model
        logging.info(f"fine_tune={args.fine_tune}: Creating LDM model based on config to train from scratch: {config}.")
        diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
        # logging.info(f"Now using {DDPMScheduler.__class__.__name__} from ldm config rather "
        #         f"than {scheduler.__class__.__name__} from {args.source_model}. "
        #         f"Please change config or code if this is not desired.")
        scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))

    if args.cond_on_acq_times:
        adjust_class_embedding(model=diffusion)

    logging.info(f"Loading text_encoder model: CLIPTestModel from {args.source_model}.")
    text_encoder = CLIPTextModel.from_pretrained(args.source_model, subfolder="text_encoder")

    logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    if torch.cuda.device_count() > 1:
        stage1 = torch.nn.DataParallel(stage1)
        diffusion = torch.nn.DataParallel(diffusion)
        text_encoder = torch.nn.DataParallel(text_encoder)

    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
    text_encoder = text_encoder.to(device)

    optimizer = optim.AdamW(diffusion.parameters(), lr=config["ldm"]["base_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0

    #checkpoint_path = str(run_dir / args.checkpoint_name)
    if resume and args.is_resumed:
        logging.info(f"Using checkpoint from {checkpoint_path} to continue training.")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        diffusion.load_state_dict(checkpoint["diffusion"])
        # Issue loading optimizer https://github.com/pytorch/pytorch/issues/2830
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # if the learning rate has changed between the training runs, we need to adjust the lr of the loaded optimizer
        if hasattr(optim, "param_groups"):
            for g in optim.param_groups:
                g['lr'] = config["ldm"]["base_lr"]
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    elif resume and not args.is_resumed:
        logging.info(f"is_resumed flag was not set and therefore checkpoint '{checkpoint_path}' is not used. Training from scratch now...")
    else:
        logging.info(f"No checkpoint found. Looked for checkpoint in {checkpoint_path}. Training from scratch now...")

    # Learning Rate Scheduler
    # We use a constant one: https://www.reddit.com/r/StableDiffusion/comments/yd56cy/dreambooth_i_compared_all_learning_rate/?rdt=37299
    lr_scheduler = None
    if args.lr_warmup_steps is not None and args.lr_warmup_steps > 0:
        lr_scheduler = get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps, # Number of steps for the warmup in the lr scheduler
            num_training_steps=len(train_loader)*args.n_epochs, # Total number of training steps. TODO: If resume, this could be adjusted.
        )


    # Scale Factor:
    # Define the vae scaling factor that will be used to (a) multiply vae-encoded latents and (b) divide latents before vae-decoding
    # scaling factor in SD is derived from the length of block_out_channels of vae
    # See code here: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L247C15-L247C83
    # See vae block_out_channels in SD-2-1-base config: https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/vae/config.json#L5
    if args.scale_factor is None:
        if hasattr(stage1, 'config') and stage1.config is not None and stage1.config.block_out_channels is not None:
            len_block_out_channels = len(stage1.config.block_out_channels)
        elif hasattr(stage1, 'num_channels') and stage1.num_channels is not None:
            len_block_out_channels = len(stage1.num_channels)
        else:
            len_block_out_channels = 4
        args.scale_factor = 2 ** (len_block_out_channels - 1)
        logging.info(f"Adjusted args.scale_factor for VAE latents to {args.scale_factor}")

    # Train model
    logging.info(f"Starting Training")
    val_loss = train_ldm(
        model=diffusion,
        stage1=stage1,
        scheduler=scheduler,
        text_encoder=text_encoder,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        scale_factor=args.scale_factor,
        eval_first=False, #True, #False, #True,  # True,
        cond_on_acq_times = args.cond_on_acq_times,
        clip_grad_norm_by=args.clip_grad_norm_by,
        clip_grad_norm_or_value=args.clip_grad_norm_or_value,
        snr_gamma=args.snr_gamma,
        lr_scheduler=lr_scheduler,
    )

    #log_mlflow(
    #    model=diffusion,
    #    config=config,
    #    args=args,
    #    experiment=args.experiment,
    #    run_dir=run_dir,
    #    val_loss=val_loss,
    #)


if __name__ == "__main__":
    args = parse_args()
    main(args)
