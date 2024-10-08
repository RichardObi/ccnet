""" Training script for the autoencoder with KL regulization. """
import argparse
import warnings
from pathlib import Path
import logging

import torch
import torch.optim as optim
from torch import nn
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_aekl
from util import get_dataloader, log_mlflow, setup_logger

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--adv_start", type=int, default=25, help="Epoch when the adversarial training starts.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")
    parser.add_argument("--is_resumed", action="store_true" , help="resume training from checkpoint in run_dir folder, if available..")
    parser.add_argument("--torch_detect_anomaly", type=int, default=0 , help="Detects NaN/Infs in the backward pass. Can slow down training significantly!")
    parser.add_argument("--fine_tune", action="store_true" , help="Fine tune the model. If false, the model is trained from scratch.")
    parser.add_argument("--early_stopping_after_num_epochs", type=int, default=99999, help="Stop training after this number of epochs passed without val loss improvement.")
    parser.add_argument("--clip_grad_norm_by", type=float,  default=None, help="Clip the gradient norm by this floating point value (default in torch is =2). SD training has produced NaNs when grad_norm > 2")
    parser.add_argument("--clip_grad_norm_or_value", type=str,  default='value', help="Clip either the norm of the gradients or the value of the gradients. Norm keeps same direction while value can change direction. Default is 'value'.")
    parser.add_argument("--img_width", type=int,  default=512, help="The image width that the dataloader will resize the input images to")
    parser.add_argument("--img_height", type=int,  default=512, help="The image height that the dataloader will resize the input images to")
    parser.add_argument("--weight_loss_with_mask_by", type=int, default=None, help="The pixel-wise weighting of the mse loss of the ldm (e.g. based on mask of segmented breast).") #" \

    args = parser.parse_args()
    return args

class Stage1Wrapper(nn.Module):
    """Wrapper for stage 1 model as a workaround for stacking graysacle images."""

    def __init__(self, model: nn.Module, stack_input_to_rgb:bool =False) -> None:
        super().__init__()
        self.model = model
        self.stack_input_to_rgb = stack_input_to_rgb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1 and self.stack_input_to_rgb:
            x = torch.cat((x, x, x), dim=1)
        return self.model.forward(x)

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
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    setup_logger(run_dir / f"train_aekl_{args.experiment}.log")

    logging.info(f"Run directory: {str(run_dir)}")
    logging.info(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    logging.info("Getting data...")
    cache_dir = output_dir / "cached_data_aekl"
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        img_width=args.img_width,
        img_height=args.img_height,
        model_type="autoencoder",
    )

    logging.info("Creating model...")
    config = OmegaConf.load(args.config_file)
    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])

    # Load Autoencoder to produce the latent representations
    if args.fine_tune:
        # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
        logging.info(f"fine_tune={args.fine_tune}: Loading pretrained stage 1 model from {args.source_model} vae")
        model = AutoencoderKL.from_pretrained(args.source_model, subfolder="vae") # model needs 3-channel rgb inputs
        model = Stage1Wrapper(model=model, stack_input_to_rgb=True)
    else:
        logging.info(f"fine_tune={args.fine_tune}: Initializing AutoencoderKL model from config to train from scratch.")
        model = AutoencoderKL(**config["stage1"]["params"])

    logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    optimizer_g = optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["stage1"]["disc_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    checkpoint_path = str(run_dir / "checkpoint_best_val_loss_e10.pth") # checkpoint.pth
    if resume and args.is_resumed:
        logging.info(f"Using checkpoint from {checkpoint_path} to continue training.")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"], map_location=torch.device(device))
        discriminator.load_state_dict(checkpoint["discriminator"], map_location=torch.device(device))
        optimizer_g.load_state_dict(checkpoint["optimizer_g"], map_location=torch.device(device))
        optimizer_d.load_state_dict(checkpoint["optimizer_d"], map_location=torch.device(device))

        # Issue loading optimizer https://github.com/pytorch/pytorch/issues/2830
        optimizer_g.load_state_dict(checkpoint["optimizer"], map_location=torch.device(device))
        for state in optimizer_g.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        optimizer_d.load_state_dict(checkpoint["optimizer"], map_location=torch.device(device))
        for state in optimizer_d.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    elif resume and not args.is_resumed:
        logging.info(f"is_resumed flag was not set and therefore checkpoint '{checkpoint_path}' is not used. Training from scratch now...")
    else:
        logging.info(f"No checkpoint found. Looked for checkpoint in {checkpoint_path}. Training from scratch now...")

    # We want to log realtime ML flow information accessible during training and therefor log already now the current model.
    logging.info(f"Starting MLFLOW logging. Logging model from epoch: {start_epoch}, best_loss: {best_loss}")
    log_mlflow(
        model=model,
        config=config,
        args=args,
        experiment=args.experiment,
        run_dir=run_dir,
        val_loss=best_loss,
        current_epoch = start_epoch,
    )

    # Train model
    logging.info(f"Starting Training, start_epoch: {start_epoch}, best_loss: {best_loss}")
    val_loss = train_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        kl_weight=config["stage1"]["kl_weight"],
        adv_weight=config["stage1"]["adv_weight"],
        perceptual_weight=config["stage1"]["perceptual_weight"],
        adv_start=args.adv_start,
        early_stopping_after_num_epochs = args.early_stopping_after_num_epochs,
        clip_grad_norm_by=args.clip_grad_norm_by,
        clip_grad_norm_or_value=args.clip_grad_norm_or_value,
        eval_first=False, #True,
        weight_loss_with_mask_by=args.weight_loss_with_mask_by,
    )

    log_mlflow(
        model=model,
        config=config,
        args=args,
        experiment=args.experiment,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
