""" Training script for the controlnet model in the latent space of the pretrained AEKL model. """
import argparse
import warnings
from pathlib import Path
import logging

import time
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from generative.networks.nets import ControlNet
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_controlnet
from transformers import CLIPTextModel
from util import get_dataloader, log_mlflow, setup_logger, adjust_class_embedding
from diffusers.optimization import get_scheduler
try:
    from diffusers import AutoencoderKL
except: # exception depends on diffusers library version
    from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers import UNet2DConditionModel, ControlNetModel, PNDMScheduler

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--stage1_uri", help="Path readable by load_model.")
    parser.add_argument("--ddpm_uri", help="Path readable by load_model.")
    parser.add_argument("--scale_factor", type=float, default=None, help="Path readable by load_model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")
    parser.add_argument("--is_resumed", action="store_true" , help="resume training from checkpoint in run_dir folder, if available..")
    parser.add_argument("--torch_detect_anomaly", type=int, default=0 , help="Detects NaN/Infs in the backward pass. Can slow down training significantly!")
    parser.add_argument("--use_pretrained", type=int, default=0, help="use a pretrained stage1 autoencoder instead of the trained checkpoint. 0 = False, 1 = stage 1 only, 2=stage 1 and diffusion model")
    parser.add_argument("--source_model", type=str, default="stabilityai/stable-diffusion-2-1-base", help="source model for the stage1 autoencoder and text_encoder") #" \
    parser.add_argument("--is_stage1_fine_tuned", action="store_true" , help="Info if the stage1 model was fine tuned, therefore requiring different ldm input and output dims.")
    parser.add_argument("--is_ldm_fine_tuned", action="store_true" , help="Info if the ldm model was fine tuned, therefore requiring different controlnet input and output dims.")
    parser.add_argument("--cond_on_acq_times", action="store_true" , help="If true, MRI acquisition times will be passed as conditional into controlnet during train and eval.") #" \
    parser.add_argument("--clip_grad_norm_by", type=float,  default=None, help="Clip the gradient norm by this floating point value (default in torch is =2). SD training has produced NaNs when grad_norm > 2")
    parser.add_argument("--clip_grad_norm_or_value", type=str,  default='value', help="Clip either the norm of the gradients or the value of the gradients. Norm keeps same direction while value can change direction. Default is 'value'.")
    parser.add_argument("--checkpoint_name", type=str,  default="controlnet_best_model.pth", help="The checkpoint file name and extension.")
    parser.add_argument("--img_width", type=int,  default=512, help="The image width that the dataloader will resize the input images to")
    parser.add_argument("--img_height", type=int,  default=512, help="The image height that the dataloader will resize the input images to")
    parser.add_argument("--init_from_unet", action="store_true" , help="If true, the controlnet will be initialized from the unet of the diffusion model. Otherwise, the controlnet will be initialized from monai generative models.")
    parser.add_argument("--controlnet_conditioning_scale", type=float,  default=1.0, help="The scaling factor for the conditioning output to the controlnet. This is used to scale the output of the controlnet. Default is 1.0.")
    parser.add_argument("--use_default_report_text", action="store_true" , help="If true, the default report text will be used for all samples returned from the dataloader. Otherwise, a custom report text will be loaded from the from the dataloader.")
    parser.add_argument("--snr_gamma", type=float, default=None, help="The gamma value for the signal-to-noise ratio (SNR) calculation. Recommendation is 5.0") # 5 is recommended in https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L355
    parser.add_argument("--lr_warmup_steps", type=int, default=None, help="Number of steps for the warmup in the lr scheduler. Recommendation is 500. ") # 500 is recommended in https://github.com/huggingface/diffusers/blob/ad310af0d65d5a008401ebde806bed413156cf82/examples/text_to_image/train_text_to_image.py#L349
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
    device = torch.device("cuda")

    run_dir = output_dir / args.run_dir
    if Path(run_dir).exists() and Path(run_dir / args.checkpoint_name).exists():
        resume = True
        checkpoint_path = str(run_dir / args.checkpoint_name)
    elif Path(args.checkpoint_name).exists():
        resume = True
        checkpoint_path = Path(args.checkpoint_name)
    else:
        resume = False
        checkpoint_path = Path(args.checkpoint_name)
        run_dir.mkdir(exist_ok=True)

    setup_logger(run_dir / f"train_controlnet_{args.experiment}.log")

    logging.info(f"Run directory: {str(run_dir)}")
    logging.info(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    logging.info("Getting data...")
    cache_name = "cached_data_controlnet_p1" if "P1" in str(run_dir) else "cached_data_controlnet"
    cache_dir = output_dir / cache_name
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        img_width=args.img_width,
        img_height=args.img_height,
        model_type="controlnet",
        use_default_report_text=args.use_default_report_text,
    )

    # Loading the config
    config = OmegaConf.load(args.config_file)

    stage1 = None
    diffusion = None
    if args.use_pretrained == 1:
        # Load Autoencoder to encode the images to latent representations
        # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
        logging.info(f"use_pretrained={args.use_pretrained}: Loading only pretrained stage 1 model from "
                     f"{args.source_model} -> vae. For the LDM, the one we trained ourselves will be used.")
        stage1 = AutoencoderKL.from_pretrained(args.source_model, subfolder="vae") # model needs 3-channel rgb inputs
        stage1 = Stage1Wrapper(model=stage1, stack_input_to_rgb=True)
        logging.info(f"Adjusting in_channels of controlnet ({config['controlnet']['params']['in_channels']}) and ldm "
                     f"{config['ldm']['params']['in_channels']} in config to 4 to make it compatible with PRETRAINED "
                     f"stable diffusion (2-1, xl) VAE outputs.")
        # 4 in and out embedding channels are standard in stability-ai's 2-1 and xl ldm models.
        config["controlnet"]["params"]["in_channels"] = 4
        config["ldm"]["params"]["in_channels"] = 4
        config["ldm"]["params"]["out_channels"] = 4
    elif args.use_pretrained == 2:
        logging.info(f"use_pretrained={args.use_pretrained}: Loading both pretrained stage 1 model AND "
                     f"pretrained ldm from {args.source_model}")
        stage1 = AutoencoderKL.from_pretrained(args.source_model, subfolder="vae") # model needs 3-channel rgb inputs
        stage1 = Stage1Wrapper(model=stage1, stack_input_to_rgb=True)
        # Choice of diffusion and scheduler is based on https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/model_index.json
        # unet: https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/unet/config.json
        # Note, if args.source_model is changed, the diffusion and scheduler classed might need to be changed as well.
        diffusion = UNet2DConditionModel.from_pretrained(args.source_model, subfolder="unet")
        # We load the scheduler for the pretrained ldm with which it was trained. This scheduler might differ from the one used during manual training of the ldm.
        scheduler = PNDMScheduler.from_pretrained(args.source_model, subfolder="scheduler") # https://arxiv.org/abs/2202.09778
        if scheduler.prk_timesteps is None:
            # To avoid bug when invoking len on NoneType scheduler.prk_timesteps in line 256 of diffusers/schedulers/scheduling_pndm.py,
            # we set the prk_timesteps to an empty array. prk steps are documented to be not useful for stable diffusion in
            # https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/schedulers/scheduling_pndm.py#L204-L206
            #import numpy as np
            #scheduler.prk_timesteps = np.array([]) #
            #scheduler.config.skip_prk_steps = True
            logging.info(f"Now using {DDPMScheduler.__class__.__name__} from ldm config rather "
                         f"than {scheduler.__class__.__name__} from {args.source_model}. "
                         f"Please change config or code if this is not desired.")
            scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
        logging.info(f"Adjusting controlnet config to work with PRETRAINED diffusion model ({args.source_model}). "
                     f"in_channels from {config['controlnet']['params']['in_channels']}) to 4,"
                     f"num_channels from {config['controlnet']['params']['num_channels']}) to {[320, 640, 1280]},"
                     f"num_head_channels from {config['controlnet']['params']['num_head_channels']}) to {[0, 640, 1280]},")
        # 4 in and out embedding channels are standard in stability-ai's 2-1 and xl ldm models.
        config["controlnet"]["params"]["in_channels"] = 4
        config["controlnet"]["params"]["num_channels"] = [320, 640, 1280]
        config["controlnet"]["params"]["num_head_channels"] = [0, 640, 1280]
    if stage1 is None:
        if args.is_stage1_fine_tuned:
            logging.info(
                f"Adjusting in_channels of controlnet ({config['controlnet']['params']['in_channels']}) and ldm "
                f"{config['ldm']['params']['in_channels']} in config to 4 to make it compatible with FINE_TUNED "
                f"stable diffusion (2-1, xl) VAE outputs.")
            # 4 in and out embedding channels are standard in stability-ai's 2-1 and xl ldm models.
            config["controlnet"]["params"]["in_channels"] = 4
            config["ldm"]["params"]["in_channels"] = 4
            config["ldm"]["params"]["out_channels"] = 4
        logging.info(f"Loading stage 1 model from {args.stage1_uri}")

        if ".pt" in args.stage1_uri:
            logging.info(f"args.stage1_uri ({args.stage1_uri}) points to a ckpt file. Loading ckpt now..")
            stage1 = AutoencoderKL(**config["stage1"].get("params", dict()))
            try:
                ckpt = torch.load(args.stage1_uri, map_location=torch.device(device))
                logging.debug(f"Loaded vae checkpoint file from {args.stage1_uri}. Contents: {ckpt.keys()}")
                stage1.load_state_dict(ckpt["state_dict"])
            except Exception as e1:
                logging.debug(
                    f"Could not load state_dict of the vae model using ckpt[state_dict]. Trying to load the state dict from {args.stage1_uri} directly. {e1}")
                stage1.load_state_dict(torch.load(args.ddpm_uri, map_location=torch.device(device)))
        else:
            stage1 = mlflow.pytorch.load_model(args.stage1_uri)
        stage1 = Stage1Wrapper(model=stage1)

    if diffusion is None:
        if args.is_ldm_fine_tuned:
            logging.info(f"Adjusting controlnet config to work with FINE_TUNED diffusion model ({args.source_model}). "
                         f"in_channels from {config['controlnet']['params']['in_channels']}) to 4,"
                         f"num_channels from {config['controlnet']['params']['num_channels']}) to {[320, 640, 1280]},"
                         f"num_head_channels from {config['controlnet']['params']['num_head_channels']}) to {[0, 640, 1280]},")
            # 4 in and out embedding channels are standard in stability-ai's 2-1 and xl ldm models.
            config["controlnet"]["params"]["in_channels"] = 4
            config["controlnet"]["params"]["num_channels"] = [320, 640, 1280]
            config["controlnet"]["params"]["num_head_channels"] = [0, 640, 1280]
            # we follow norm_eps as in https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/unet/config.json#L32:
            config["controlnet"]["params"]["norm_eps"] = 1e-5

        if ".pt" in args.ddpm_uri:
            logging.info(f"args.ddpm_uri ({args.ddpm_uri}) points to a ckpt file. Loading this diffusion ckpt now..")
            # Diffusion model can be either of type UNet2DConditionModel (diffusers) or DiffusionModelUNet (monai). We test loading any of both types here.
            diffusion_models = [DiffusionModelUNet(**config["ldm"].get("params", dict())), UNet2DConditionModel.from_pretrained(args.source_model, subfolder="unet"),]
            diffusion = None
            for diffusion_model in diffusion_models:
                if args.cond_on_acq_times:
                    # We need to adjust the network architecture accordingly before loading the state_dict
                    adjust_class_embedding(model=diffusion_model)
                try:
                    ckpt = torch.load(args.ddpm_uri, map_location=torch.device(device))
                    logging.debug(f"Loaded ldm checkpoint file from {args.ddpm_uri}. Contents: {ckpt.keys()}")
                    diffusion_model.load_state_dict(ckpt["diffusion"])
                except Exception as e1:
                    logging.info(
                        f"Could not load state_dict of the diffusion model using ckpt[diffusion]. Trying to load the state dict from {args.ddpm_uri} directly. Exception: {e1}")
                    #time.sleep(20)
                    try:
                        diffusion_model.load_state_dict(torch.load(args.ddpm_uri, map_location=torch.device(device)))
                    except Exception as e2:
                        logging.info(
                            f"Could not load state_dict directly from {args.ddpm_uri} directly. Exception: {e2}")
                        continue
                diffusion = diffusion_model
                break # if we found a model that works, we break the loop
            if diffusion is None:
                raise ValueError(f"Could not load the diffusion model from {args.ddpm_uri}.")
        else:
            diffusion = mlflow.pytorch.load_model(args.ddpm_uri)
        # get the standard diffusion scheduler from the config
        # TODO: Here we should check which scheduler was used during training of the ldm and use that one. For now DDPM is the only one available.
        scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))

    # only training the controlnet, not stage1 or diffusion
    stage1.eval()
    diffusion.eval()

    # Init the controlnet based on the config params
    logging.info(f"Loading controlnet model from config {config}.")
    #if args.use_pretrained == 2:

    #    controlnet = ControlNetModel(**config["controlnet"].get("params", dict()))
    #else:
    if args.init_from_unet:
        # Init the controlnet of the diffusers library

        # First, to init from unet, we need to remove the config.num_class_embeds potentially present in diffusion unet.
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py#L509
        if diffusion is not None and diffusion.config is not None and diffusion.config.num_class_embeds is not None:
            previous_num_class_embeds = diffusion.config.num_class_embeds
            diffusion.config.num_class_embeds = None # this won't change the architecture/weights of the diffusion model as it is already initialized

        # https://huggingface.co/docs/diffusers/v0.24.0/en/api/models/controlnet#diffusers.ControlNetModel
        # https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/controlnet.py#L439C9-L439C18
        controlnet = ControlNetModel.from_unet(unet=diffusion,
                                               #conditioning_embedding_out_channels = (64, 128, 128, 256) #https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py#L80
                                               conditioning_channels=1, # 3 # we need to define if we use rgb here or grayscale
                                               controlnet_conditioning_channel_order="rgb") # needs to be rgb or bgr, for grayscale it should still be fine to use 'rgb' as it seems input images is not adjusted in this case.

        if args.cond_on_acq_times:
            diffusion.config.num_class_embeds = previous_num_class_embeds
            # TODO: Potential ablation - do we also need the class embeddings in the controlnet?
            # the acq time input linear layers are initialized from scratch (not from ldm) so they need to be
            # learned again (which could be an advantage or disadvantage. TODO: Possible ablation)
            adjust_class_embedding(model=controlnet)
    else:
        # Init the controlnet of the monai library
        # https://docs.monai.io/en/latest/networks.html#controlnet
        controlnet = ControlNet(**config["controlnet"].get("params", dict()))

        if args.cond_on_acq_times:
            # TODO: Potential ablation - do we also need the class embeddings in the controlnet?
            adjust_class_embedding(model=controlnet)

        # Copy weights from the DM to the controlnet
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    # Freeze the weights of the diffusion model. We only want to train the controlnet.
    for p in diffusion.parameters():
        p.requires_grad = False

    logging.info(f"Loading text_encoder model: CLIPTestModel from {args.source_model}.")
    text_encoder = CLIPTextModel.from_pretrained(args.source_model, subfolder="text_encoder")

    logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")

    if torch.cuda.device_count() > 1:
        stage1 = torch.nn.DataParallel(stage1)
        diffusion = torch.nn.DataParallel(diffusion)
        controlnet = torch.nn.DataParallel(controlnet)
        text_encoder = torch.nn.DataParallel(text_encoder)

    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
    controlnet = controlnet.to(device)
    text_encoder = text_encoder.to(device)

    optimizer = optim.AdamW(controlnet.parameters(), lr=config["controlnet"]["base_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0

    #checkpoint_path = str(run_dir / args.checkpoint_name)
    if resume:
        logging.info(f"Using checkpoint from {checkpoint_path} to continue training.")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        controlnet.load_state_dict(checkpoint["controlnet"])
        # Issue loading optimizer https://github.com/pytorch/pytorch/issues/2830
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
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
        if stage1.config is not None and stage1.config.block_out_channels is not None:
            len_block_out_channels = len(stage1.config.block_out_channels)
        elif stage1.num_channels is not None:
            len_block_out_channels = len(stage1.num_channels)
        else:
            len_block_out_channels = 4
        args.scale_factor = 2 ** (len_block_out_channels - 1)
        logging.info(f"Adjusted args.scale_factor for VAE latents to {args.scale_factor}")

    # Train model
    logging.info(f"Starting Training")
    val_loss = train_controlnet(
        controlnet=controlnet,
        diffusion=diffusion,
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
        eval_first= False, #True,  # True,
        cond_on_acq_times=args.cond_on_acq_times,
        clip_grad_norm_by=args.clip_grad_norm_by,
        clip_grad_norm_or_value=args.clip_grad_norm_or_value,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        snr_gamma=args.snr_gamma,
        lr_scheduler=lr_scheduler,
    )

    #log_mlflow(
    #    model=controlnet,
    #    config=config,
    #    args=args,
    #    experiment=args.experiment,
    #    run_dir=run_dir,
    #    val_loss=val_loss,
    #)


if __name__ == "__main__":
    args = parse_args()
    main(args)
