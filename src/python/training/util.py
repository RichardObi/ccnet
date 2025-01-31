"""Utility functions for training."""


from pathlib import Path
from typing import Tuple, Union
import logging
import sys
import datetime

from PIL import Image
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from custom_transforms import ApplyTokenizerd
from mlflow import start_run
from monai import transforms
from monai.data import Dataset, PersistentDataset, CacheDataset
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import InterpolationMode

import mlflow.pytorch
from generative.networks.nets import DiffusionModelUNet, ControlNet
from generative.networks.schedulers import DDPMScheduler
try:
    from diffusers import AutoencoderKL
except: # exception depends on diffusers library version
    from diffusers.models.autoencoder_kl import AutoencoderKL # pretrained model
from diffusers import UNet2DConditionModel, ControlNetModel, PNDMScheduler # pretrained model



# ----------------------------------------------------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------------------------------------------------
FILENAME = 'filename'
FILENAME_TARGET = 'filename_target'
REPORT = 'report'
REPORT_RAW = 'report_raw'
SOURCE = 'source'
TARGET = 'target'
PATIENT_ID = 'patient_id'
ACQ_TIME = 'acquisition_time'
SEGMENTATION_MASK = 'segmentation_mask'

def get_datalist(
    ids_path: str,
    default_report_text: str = "dynamic contrast-enhanced magnetic resonance image of a breast.",
    use_default_report_text: bool = False,
    upper_limit: int | None = None,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    if upper_limit is not None:
        df = df[212:212 + int(upper_limit)]

    data_dicts = []
    for index, row in df.iterrows():
        try:
            report = f"{row[REPORT]}"
            report_raw = f"{row[REPORT_RAW]}"
            if use_default_report_text:
                raise Exception("Fallback to standard report text.")
        except Exception as e:
            #logging.warning(f"row[{REPORT}] not found, now using default report text. index={index}, row={row}. Exception: {e}")
            report =  default_report_text
            report_raw = default_report_text

        #if index%100 == 0:
        #    logging.info(f"{index}: report: {report}")

        data_dicts.append(
            {
                FILENAME: f"{row[SOURCE]}",
                FILENAME_TARGET: f"{row[TARGET]}",
                SOURCE: f"{row[SOURCE]}",
                TARGET: f"{row[TARGET]}",
                REPORT: report,
                REPORT_RAW: report_raw,
                PATIENT_ID: f"{row[PATIENT_ID]}",
                ACQ_TIME: float(row[ACQ_TIME]), # linear layer expects floats ("Half")
                SEGMENTATION_MASK: f"{row[SEGMENTATION_MASK]}",
            }
        )


    # TODO Print number of subjects, number of phases.
    logging.info(f"Found {len(data_dicts)} image pairs.")

    logging.info(
        f"Example: First row of dataset: row={data_dicts[0]}.")

    return data_dicts


def get_test_dataloader(
    batch_size: int,
    test_ids: str,
    num_workers: int = 8,
    img_height: int = 512,
    img_width: int = 512,
    upper_limit: int | None = None,
    use_default_report_text: bool = False,
):

    test_transforms = transforms.Compose(
        [
            # for guidance scale if text, we need to also pass an empty string through the controlnet and ldm
            transforms.Lambdad(
                keys=[REPORT],
                func=lambda x: x
            ),
            # see https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1012-L1014C58
            transforms.LoadImaged(keys=[SOURCE, TARGET]),
            transforms.EnsureChannelFirstd(keys=[SOURCE, TARGET]),
            transforms.Resized(keys=[SOURCE, TARGET], spatial_size=(img_height, img_width), mode='bicubic'),# ,mode=InterpolationMode.BICUBIC),
            # transforms.Rotate90d(keys=[SOURCE, TARGET], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
            # transforms.Flipd(keys=[SOURCE, TARGET], spatial_axis=1),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
            # Watch out ,the image range in diffusers stable diffusion (vae) is in range -1,1: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/image_processor.py#L140
            transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0,
                                            clip=True),
            # transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            ApplyTokenizerd(keys=[REPORT]),
            transforms.ToTensord(keys=[ACQ_TIME, SOURCE, TARGET, REPORT]),
        ]
    )

    test_dicts = get_datalist(ids_path=test_ids, upper_limit=upper_limit, use_default_report_text=use_default_report_text)
    test_ds = CacheDataset(data=test_dicts, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )
    logging.info(f"test_loader returned {len(test_loader)} items (batch size: {batch_size}). "
                 f"Here first test_loader batch as example: {next(iter(test_loader))}")

    return test_loader


def get_dataloader(
    cache_dir: Union[str, Path],
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    num_workers: int = 8,
    img_height: int = 512,
    img_width: int = 512,
    model_type: str = "autoencoder",
    use_default_report_text: bool = False,
):
    # Define transformations

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=[SOURCE, TARGET]),
            transforms.EnsureChannelFirstd(keys=[SOURCE, TARGET]),
            transforms.Resized(keys=[SOURCE, TARGET], spatial_size=(img_height, img_width), mode='bicubic'), #,mode=InterpolationMode.BICUBIC),
            ##transforms.Rotate90d(keys=[SOURCE, TARGET], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
            #transforms.Flipd(keys=[TARGET], spatial_axis=1),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
            # Watch out ,the image range in diffusers stable diffusion (vae) is in range -1,1: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/image_processor.py#L140
            transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
            #transforms.ScaleIntensityRanged(keys=[TARGET], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            ApplyTokenizerd(keys=[REPORT]),
            transforms.ToTensord(keys=[ACQ_TIME, SOURCE, TARGET, REPORT]),
        ]
    )
    if model_type == "autoencoder":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=[TARGET]),
                transforms.EnsureChannelFirstd(keys=[TARGET]),
                transforms.Resized(keys=[TARGET], spatial_size=(img_height, img_width), mode='bicubic'),
                # ,mode=InterpolationMode.BICUBIC),
                #transforms.Rotate90d(keys=[TARGET], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
                #transforms.Flipd(keys=[TARGET], spatial_axis=1),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
                # Watch out ,the image range in diffusers stable diffusion (vae) is in range -1,1: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/image_processor.py#L140
                transforms.ScaleIntensityRanged(keys=[TARGET], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
                #transforms.ScaleIntensityRanged(keys=[TARGET], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),

                ## Data augmentations
                transforms.RandFlipd(keys=[TARGET], prob=0.1, spatial_axis=0), # prob from 0.5 to 0.1 as test/val breast mri slices will always have same orientation.
                transforms.RandAffined(
                    keys=[TARGET],
                    translate_range=(2, 2), #translate_range=(-2, 2),
                    scale_range=(0.05, 0.05), #scale_range=(-0.05, 0.05),
                    # spatial_size=[160, 224], # Outcomment to use default spatial size of input img, https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandAffine
                    prob=0.25,
                ), # prob from 0.5 to 0.25 to reduce amount of augmentation
                transforms.RandShiftIntensityd(keys=[TARGET], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=[TARGET], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=[TARGET], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=[TARGET], threshold=0, above=True, cval=0.0),
                transforms.ToTensord(keys=[TARGET]),
            ]
        )
    if model_type == "diffusion":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=[SOURCE, TARGET]),
                transforms.EnsureChannelFirstd(keys=[SOURCE, TARGET]),
                #transforms.Rotate90d(keys=[SOURCE, TARGET], k=-1, spatial_axes=(0, 1)),
                # Fix flipped image read # Outcommented, as images should already be in correct orientation.
                transforms.Resized(keys=[SOURCE, TARGET], spatial_size=(img_height, img_width), mode='bicubic'), #,mode=InterpolationMode.BICUBIC),
                # transforms.Flipd(keys=[SOURCE, TARGET], spatial_axis=1),  # Fix flipped image read Fix flipped image read # Outcommented, as images should already be in correct orientation.
                # Watch out ,the image range in diffusers stable diffusion (vae) is in range -1,1: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/image_processor.py#L140
                transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0,
                                                clip=True),
                # transforms.ScaleIntensityRanged(keys=[TARGET], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),

                ## Data augmentations
                transforms.RandFlipd(keys=[SOURCE, TARGET], prob=0.1, spatial_axis=0),
                # prob from 0.5 to 0.1 as test/val breast mri slices will always have same orientation.
                transforms.RandAffined(
                    keys=[SOURCE, TARGET],
                    translate_range=(2, 2),  # translate_range=(-2, 2),
                    scale_range=(0.01, 0.01),  # scale_range=(-0.01, 0.01),
                    # spatial_size=[160, 224], # Outcomment to use default spatial size of input img, https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandAffine
                    prob=0.10,
                ),  # prob from 0.25 to 0.10 to reduce amount of augmentation
                transforms.RandShiftIntensityd(keys=[SOURCE, TARGET], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=[SOURCE, TARGET], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=[SOURCE, TARGET], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=[SOURCE, TARGET], threshold=-1, above=True, cval=-1.0),
                ApplyTokenizerd(keys=[REPORT]),
                transforms.RandLambdad(
                    keys=[REPORT],
                    prob=0.10,
                    func=lambda x: torch.cat(
                        (49406 * torch.ones(1, 1), 49407 * torch.ones(1, x.shape[1] - 1)), 1
                    ).long(),
                ),  # 49406: BOS token 49407: PAD token
                transforms.ToTensord(keys=[ACQ_TIME, SOURCE, TARGET, REPORT]),
            ]
        )
    if model_type == "controlnet":
        val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=[SOURCE, TARGET]),
                transforms.EnsureChannelFirstd(keys=[SOURCE, TARGET]),
                transforms.Resized(keys=[SOURCE, TARGET], spatial_size=(img_height, img_width), mode='bicubic'),
                # ,mode=InterpolationMode.BICUBIC),
                #transforms.Rotate90d(keys=[SOURCE, TARGET], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
                #transforms.Flipd(keys=[SOURCE, TARGET], spatial_axis=1),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
                # Watch out ,the image range in diffusers stable diffusion (vae) is in range -1,1: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/image_processor.py#L140
                transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
                #transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
                ApplyTokenizerd(keys=[REPORT]),
                transforms.ToTensord(keys=[ACQ_TIME, SOURCE, TARGET, REPORT]),
            ]
        )
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=[SOURCE, TARGET]),
                transforms.EnsureChannelFirstd(keys=[SOURCE, TARGET]),
                transforms.Resized(keys=[SOURCE, TARGET], spatial_size=(img_height, img_width), mode='bicubic'), #,mode=InterpolationMode.BICUBIC),
                #transforms.Rotate90d(keys=[SOURCE, TARGET], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
                #transforms.Flipd(keys=[SOURCE, TARGET], spatial_axis=1),  # Fix flipped image read # Outcommented, as images should already be in correct orientation.
                # Watch out ,the image range in diffusers stable diffusion (vae) is in range -1,1: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/image_processor.py#L140
                transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0, clip=True),
                #transforms.ScaleIntensityRanged(keys=[SOURCE, TARGET], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0,clip=True),

                ## Data augmentations
                transforms.RandFlipd(keys=[SOURCE, TARGET], prob=0.1, spatial_axis=0), # prob from 0.5 to 0.1 as test/val breast mri slices will always have same orientation.
                transforms.RandAffined(
                    keys=[SOURCE, TARGET],
                    translate_range=(2, 2), #translate_range=(-2, 2),
                    scale_range=(0.01, 0.01), #scale_range=(-0.01, 0.01),
                    prob=0.10,
                    #spatial_size = [512, 512], # FIXME test
                    #spatial_size=[160, 224], # Outcomment to use default spatial size of input img, https://docs.monai.io/en/stable/transforms.html#monai.transforms.RandAffine
                ), # prob from 0.25 to 0.10 to reduce amount of augmentation
                transforms.RandShiftIntensityd(keys=[SOURCE, TARGET], offsets=0.05, prob=0.1), #0.05 where images have values between 0.0 and 1.0 (see ScaleIntensityRanged)
                transforms.RandAdjustContrastd(keys=[SOURCE, TARGET], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=[SOURCE, TARGET], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=[SOURCE, TARGET], threshold=-1, above=True, cval=-1.0),
                ApplyTokenizerd(keys=[REPORT]),
                transforms.RandLambdad(
                    keys=[REPORT],
                    prob=0.10,
                    func=lambda x: torch.cat(
                        (49406 * torch.ones(1, 1), 49407 * torch.ones(1, x.shape[1] - 1)), 1
                    ).long(),
                ),  # 49406: BOS token 49407: PAD token
                transforms.ToTensord(keys=[ACQ_TIME, SOURCE, TARGET, REPORT]),
            ]
        )

    train_dicts = get_datalist(ids_path=training_ids, use_default_report_text=use_default_report_text)
    #train_ds = Dataset(data=train_dicts, transform=train_transforms)
    train_ds = PersistentDataset(data=train_dicts, transform=train_transforms, cache_dir=str(cache_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, #True,
        num_workers=num_workers,
        drop_last=True, #drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dicts = get_datalist(ids_path=validation_ids, use_default_report_text=use_default_report_text)
    #val_ds = Dataset(data=val_dicts, transform=val_transforms)
    val_ds = PersistentDataset(data=val_dicts, transform=val_transforms, cache_dir=str(cache_dir))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # True,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )
    #logging.info(f"train_loader: {next(iter(train_loader))}")
    #logging.info(f"val_loader: {next(iter(val_loader))}")
    return train_loader, val_loader


# ----------------------------------------------------------------------------------------------------------------------
# LOGS
# ----------------------------------------------------------------------------------------------------------------------
def recursive_items(dictionary, prefix=""):
    for key, value in dictionary.items():
        if type(value) in [dict, DictConfig]:
            yield from recursive_items(value, prefix=str(key) if prefix == "" else f"{prefix}.{str(key)}")
        else:
            yield (str(key) if prefix == "" else f"{prefix}.{str(key)}", value)



def log_mlflow(
    model,
    config,
    args,
    experiment: str,
    run_dir: Path,
    val_loss: float,
    current_epoch: int = None,
):
    """Log model and performance on Mlflow system"""
    config = {**OmegaConf.to_container(config), **vars(args)}
    logging.info(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        logging.info(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        logging.info(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in recursive_items(config):
            mlflow.log_param(key, str(value))

        mlflow.log_artifacts(str(run_dir / "train"), artifact_path="events_train")
        mlflow.log_artifacts(str(run_dir / "val"), artifact_path="events_val")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model
        if current_epoch is not None:
            mlflow.pytorch.log_model(raw_model, f"non_final_model_at_epoch{current_epoch}")
        else:
            mlflow.pytorch.log_model(raw_model, "final_model")


def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor = None,
    dpi: int = 300,
    change_figure_min: bool = False,
    number_in_batch=0,
):
    # get the number of images in batch
    if img.shape[0] == 1:
        img_npy_0 = img[number_in_batch].cpu().numpy()
        recons_npy_0 = recons[number_in_batch].cpu().numpy() if recons is not None else None
    else:
        if recons is not None:
            try:
                #recons_npy_0 = np.clip(a=recons[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
                #recons_npy_1 = np.clip(a=recons[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
                recons_npy_0 = np.clip(a=recons[number_in_batch, :, :, :].cpu().numpy(), a_min=0, a_max=1)
                recons_npy_1 = np.clip(a=recons[number_in_batch+1, :, :, :].cpu().numpy(), a_min=0, a_max=1)
                img_npy_1 = np.clip(a=img[number_in_batch+1, :, :, :].cpu().numpy(), a_min=0, a_max=1)
                img_npy_0 = np.clip(a=img[number_in_batch, :, :, :].cpu().numpy(), a_min=0, a_max=1)  # can be 1 or 3 channels
            except:
                recons_npy_0 = np.clip(a=recons[number_in_batch].cpu().numpy(), a_min=-1 if change_figure_min else 0, a_max=1) # -1 to 1
                recons_npy_1 = np.clip(a=recons[number_in_batch+1].cpu().numpy(), a_min=-1 if change_figure_min else 0, a_max=1) # -1 to 1
                img_npy_1 = np.clip(a=img[number_in_batch+1].cpu().numpy(), a_min=0, a_max=1) # -1 to 1
                img_npy_0 = np.clip(a=img[number_in_batch].cpu().numpy(), a_min=0, a_max=1)  # -1 to 1
        else:
            img_npy_0 = np.clip(a=img[number_in_batch].cpu().numpy(), a_min=-1, a_max=1)  # -1 to 1

        img_npy_1 = img[number_in_batch + 1].cpu().numpy()
        img_npy_0 = img[number_in_batch].cpu().numpy()
        recons_npy_0 = recons[number_in_batch].cpu().numpy()
        recons_npy_1 = recons[number_in_batch + 1].cpu().numpy()
        # scale recons_npy_1 to 0 to 255
        #recons_npy_1 = (recons_npy_1 + 1) * 127.5 # convert to int 0-255
        #img_npy_0 = np.clip(a=img[0, 0, :, :].cpu().numpy(), a_min=0, a_max=1)
        #img_npy_1 = np.clip(a=img[1, 0, :, :].cpu().numpy(), a_min=0, a_max=1)

    if recons is not None:
        if img.shape[0] > 1:
            img_row_0 = np.concatenate(
                (
                    img_npy_0,
                    recons_npy_0,
                    img_npy_1,
                    recons_npy_1,
                ),
                axis=1,
            )
        else:
            img_row_0 = np.concatenate(
                (
                    img_npy_0,
                    recons_npy_0,
                ),
                axis=1,
            )
    else:
        img_row_0 = img_npy_0
    try:
        img_row_0 = np.transpose(img_row_0, (2, 1, 0)).astype(np.float64)
    except:
        logging.warning(f"Could not transpose(2, 1, 0) image of shape {img_row_0.shape}.")
    fig = plt.figure(dpi=dpi)
    plt.imshow(img_row_0, cmap="gray")
    plt.axis("off")
    return fig


def log_reconstructions(
    image: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    reconstruction: torch.Tensor = None,
    title: str = "RECONSTRUCTION",
    model_type: str="autoencoder",
) -> None:
    logging.info(
        f"image {title} of shape {image.shape}: Has mean {np.mean(image)} and std {np.std(image)} and min {np.min(image)} and max {np.max(image)}. ")
    fig = get_figure(
        image,
        reconstruction,
    )
    add_custom_figure(fig=fig, writer=writer, step=step, title=title, model_type=model_type)

def log_image_pairs(
    image: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    reconstruction: torch.Tensor = None,
    title: str = "IMAGE_PAIR",
    model_type: str="autoencoder",
) -> None:
    logging.info(
        f"image {title} of shape {image.shape}: Has mean {np.mean(image)} and std {np.std(image)} and min {np.min(image)} and max {np.max(image)}. ")
    fig = get_figure(
        image,
        reconstruction,
        change_figure_min=True,
        number_in_batch=0,
    )
    add_custom_figure(fig=fig, writer=writer, step=step, title=title, model_type=model_type)
    if image.shape[0] > 3:
        fig = get_figure(
            image,
            reconstruction,
            change_figure_min=True,
            number_in_batch=2,
        )
        add_custom_figure(fig=fig, writer=writer, step=step, title=title, model_type=model_type)
    if image.shape[0] > 5:
        fig = get_figure(
            image,
            reconstruction,
            change_figure_min=True,
            number_in_batch=4,
        )
        add_custom_figure(fig=fig, writer=writer, step=step, title=title, model_type=model_type)

def rgb2gray(rgb):
    # https://stackoverflow.com/a/12201744
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

@torch.no_grad()
def log_ldm_sample(
    model: nn.Module,
    stage1: nn.Module,
    text_encoder,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    scale_factor: float = 1.0,
    text_condition = None, # tokenized and tensorized text condition
    text_condition_raw = None, # original string value of text condition
    dpi: int = 300,
    class_labels = None,
    epoch: str= '',
) -> None:
    """Log a sample from the LDM model."""

    conditional_txt = "text-conditioned_" if text_condition_raw is not None else ''

    # Watch out - it is assumed that the noise distribution in the LDM training is distributed similarly as torch.rand((1,) here.
    # -> uniform distribution on the interval [0, 1) -> mean=0.5, std=1/12
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    if text_condition == None:
        prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long().to(device)
    else:
        prompt_embeds = text_condition # get only first report element from batch
    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_embeds.squeeze(1))
    prompt_embeds = prompt_embeds[0]

    if scheduler.timesteps is None:
        scheduler_timesteps = scheduler._timesteps
    else:
        scheduler_timesteps = scheduler.timesteps

    for t in tqdm(scheduler_timesteps, ncols=70):
        timestep = torch.asarray((t,)).to(device)
        with torch.no_grad():
            noise_pred = sample_from_ldm(model=model, noisy_e=latent, timesteps=timestep, prompt_embeds=prompt_embeds, class_labels=class_labels)
        try:
            latent, _ = scheduler.step(noise_pred, t, latent)
        except:
            latent = scheduler.step(noise_pred, t, latent)
        #latent = scheduler.add_noise(original_samples=noise_pred, noise=latent, timesteps=timestep)

    # Let's print some statistics of latent transformed by ldm before re-scaling and decoding.
    logging.info(
        f"Step_{step}: LDMs transformed latent z (shape={latent.shape}, after {len(scheduler_timesteps)} timesteps) "
        f"has mean {torch.mean(latent)} and std {torch.std(latent)} and min {torch.min(latent)} and max {torch.max(latent)}. "
        f"before dividing it by scale_factor ({scale_factor}).")
    with torch.no_grad():
        x_hat = stage1.model.decode(latent / scale_factor)

    try:
        x_hat_numpy = x_hat[0, :, :, :].cpu().numpy().astype(np.float64)
    except Exception as e:
        x_hat_numpy = x_hat.sample[0, :, :, :].cpu().numpy().astype(np.float64)

    logging.info(
        f"Step_{step}: x_hat_numpy (after scale factor {scale_factor} multiplication and decoding) "
        f"has mean {np.mean(x_hat_numpy)} and std {np.std(x_hat_numpy)} and min {np.min(x_hat_numpy)} and max {np.max(x_hat_numpy)}. ")
    try:
        postprocess_and_save_syn_image(x_hat=x_hat,
                                       dpi=dpi,
                                       writer=writer,
                                       step=step,
                                       conditional_txt=f"{conditional_txt}",
                                       dataset_name='val',
                                       epoch=epoch,
                                       text_condition_raw=text_condition_raw,
                                       class_labels=class_labels)
    except Exception as e:
        logging.error(f"Step: {step}. Failure while trying to postprocess_and_save_syn_image sample in {writer.logdir} : {e}")


def postprocess_and_save_syn_image(x_hat, dpi=100, writer='', step='', conditional_txt ='', dataset_name='', text_condition_raw='', epoch='', class_labels=None, enforced_name=None):
    # TODO: This is the desired output image for 2d to 3d reconstruction of MRI slices (for better intensity correspondence of 2d slices)
    ############### Interval Mapped RGB-To-Grayscale -1,1 Matplotlib Image ###############
    try:
        try:
            # saving image using matplotlib
            img_1 = x_hat[0].cpu().numpy().astype(np.float64)
        except Exception as e:
            logging.info(
                f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
            img_1 = x_hat.sample[0].cpu().numpy().astype(np.float64)
        img_1 = interval_mapping(img_1, from_min=-1., from_max=1., to_min=0., to_max=1.)
        img_1 = img_1.squeeze(0)
        try:
            img_0 = np.transpose(img_1, (1, 2, 0)).astype(np.float64)  # or (2, 1, 0)
        except Exception as e:
            logging.warning(f"Could not transpose(1, 2, 0) grayscale unchanged image of shape {img_1.shape}: {e}")
        try:
            # Assumption: Image is in RGB format
            img_1_gray = rgb2gray(img_0)
        except Exception as e:
            logging.warning(f"Could not convert RGB to grayscale image of shape {img_1.shape}: {e}")

        logging.info(
            f"rgb2gray Interval mapped image (from -1,1 to 0-1) has mean {np.mean(img_1_gray)} and std {np.std(img_1_gray)} and min {np.min(img_1_gray)} and max {np.max(img_1_gray)}. ")

        # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
        fig = plt.figure(dpi=dpi)  # dpi=300
        plt.imshow(img_1_gray, cmap="gray")
        plt.axis("off")
        if enforced_name is not None:
            add_custom_figure(fig=fig, enforced_name=enforced_name)
        else:
            add_custom_figure(fig=fig, writer=writer, step=step, title=f"{conditional_txt}INTERVAL_MAPPED_FROMm11_TO01_GRAYSCALE_SAMPLE",
                          model_type=f"ldm-{dataset_name}",
                          text_condition_raw=f"{text_condition_raw} | (acquisition_time: {class_labels.item() if class_labels is not None else 'None'})",
                          epoch=epoch)
        plt.close()
    except Exception as e:
        logging.error(f"Failure while trying to save image sample: {e}")


    # ############### Other visualisation variations ###############
    # if save_image_variations:
    ############### Clamped (0,1) Grayscale Matplotlib Image ###############
    # try:
    #     # saving image using matplotlib
    #     img_0 = torch.clamp(x_hat[0], min=0., max=1.).cpu().numpy().astype(np.float64)
    # except Exception as e:
    #     logging.debug(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
    #     img_0 = torch.clamp(x_hat.sample[0], min=0., max=1.).cpu().numpy().astype(np.float64)
    # finally:
    #     img_0 = img_0.squeeze(0)
    #     try:
    #         img_0 = np.transpose(img_0, (1, 2, 0)).astype(np.float64) # or (2, 1, 0)
    #     except Exception as e:
    #         logging.warning(f"Could not transpose(2, 1, 0) grayscale unchanged image of shape {img_0.shape}: {e}")
    #     try:
    #         img_0_gray = rgb2gray(img_0)
    #     except Exception as e:
    #         logging.warning(f"Could not convert RGB to grayscale image of shape {img_0.shape}: {e}")
    #     logging.info(f"Torch clamped Grayscale rgb2gray-transformed image (Matplotlib): Decoded output image has mean {np.mean(img_0_gray)} and std {np.std(img_0_gray)} and max {np.max(img_0_gray)} and min {np.min(img_0_gray)}.")
    #     # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #
    #     fig = plt.figure(dpi=dpi)  # dpi=300
    #     plt.imshow(img_0_gray, cmap="gray")
    #     plt.axis("off")
    #     add_custom_figure(writer, step, f"{conditional_txt}GRAYSCALE_SAMPLE_CLAMPED_01", fig, model_type="ldm", text_condition_raw=text_condition_raw)

    #     # Interval Mapped min max RGB Matplotlib Image
    #     try:
    #         try:
    #             # saving image using matplotlib
    #             img_1 = x_hat[0, :, :, :].cpu().numpy().astype(np.float64)
    #         except Exception as e:
    #             logging.info(
    #                 f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
    #             img_1 = x_hat.sample[0, :, :, :].cpu().numpy().astype(np.float64)
    #         finally:
    #             img_1 = interval_mapping(img_1, from_min=np.min(img_1), from_max=np.max(img_1), to_min=0., to_max=1.)
    #             logging.info(
    #                 f"RGB Clipped 0-1 interval mapped image has mean {np.mean(img_1)} and std {np.std(img_1)} and min {np.min(img_1)} and max {np.max(img_1)}. ")
    #             try:
    #                 img_0 = np.transpose(img_1, (1, 2, 0)).astype(np.float64)  # or (2, 1, 0)
    #             except Exception as e:
    #                 logging.warning(
    #                     f"Could not transpose(2, 1, 0) grayscale unchanged image of shape {img_1.shape}: {e}")
    #
    #             # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #             fig = plt.figure(dpi=dpi)  # dpi=300
    #             plt.imshow(img_1)
    #             plt.axis("off")
    #             add_custom_figure(writer, step, f"{conditional_txt}INTERVAL_MAPPED_01_RGB_SAMPLE", fig,
    #                               model_type="ldm", text_condition_raw=text_condition_raw)
    #             plt.close()
    #     except Exception as e:
    #         logging.error(f"Failure while trying to save RGB sample: {e}")
    #
    #
    #     # Clipped Grayscale -1 to 1  Matplotlib Image
    #     try:
    #         # saving image using matplotlib
    #         img_5 = np.clip(a=x_hat[0, 0, :, :].cpu().numpy().astype(np.float64), a_min=-1, a_max=1)
    #     except Exception as e:
    #         logging.info(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
    #         img_5 = np.clip(a=x_hat.sample[0, 0, :, :].cpu().numpy().astype(np.float64), a_min=-1, a_max=1)
    #     finally:
    #         logging.info(
    #             f"Grayscale-1 to 0:x_hat has mean {np.mean(img_5)} and std {np.std(img_5)} and min {np.min(img_5)} and max {np.max(img_5)}. ")
    #         # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #         fig = plt.figure(dpi=dpi)  # dpi=300
    #         plt.imshow(img_5, cmap="gray")
    #         plt.axis("off")
    #         add_custom_figure(writer, step, f"{conditional_txt}Clipped_m11_grayscale_SAMPLE", fig, model_type="ldm",
    #                           text_condition_raw=text_condition_raw)
    #         plt.close()
    #
    #     # Interval mapped Grayscale -1 to 1  Matplotlib Image
    #     try:
    #         # saving image using matplotlib
    #         img_7 = x_hat[0, 0, :, :].cpu().numpy().astype(np.float64)
    #     except Exception as e:
    #         logging.info(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
    #         img_7 = x_hat.sample[0, 0, :, :].cpu().numpy().astype(np.float64)
    #     finally:
    #         img_6 = interval_mapping(img_7, from_min=np.min(img_7), from_max=np.max(img_7), to_min=-1, to_max=1)
    #         logging.info(
    #             f"Interval_Mapped image -1 to 1: x_hat has mean {np.mean(img_6)} and std {np.std(img_6)} and min {np.min(img_6)} and max {np.max(img_6)}. ")
    #         # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #         fig = plt.figure(dpi=dpi)  # dpi=300
    #         plt.imshow(img_6, cmap="gray")
    #         plt.axis("off")
    #         add_custom_figure(writer, step, f"{conditional_txt}Interval_Mapped_m11_grayscale_SAMPLE", fig, model_type="ldm",
    #                           text_condition_raw=text_condition_raw)
    #         plt.close()
    #         img_6 = interval_mapping(img_7, from_min=np.min(img_7), from_max=np.max(img_7), to_min=0, to_max=1)
    #         logging.info(
    #             f"Interval_Mapped image 0 to 1: x_hat has mean {np.mean(img_6)} and std {np.std(img_6)} and min {np.min(img_6)} and max {np.max(img_6)}. ")
    #         # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #         fig = plt.figure(dpi=dpi)  # dpi=300
    #         plt.imshow(img_6, cmap="gray")
    #         plt.axis("off")
    #         add_custom_figure(writer, step, f"{conditional_txt}Interval_Mapped_01_grayscale_SAMPLE", fig, model_type="ldm",
    #                           text_condition_raw=text_condition_raw)
    #         plt.close()
    #
    #     # Clipped RGB Matplotlib Image
    #     try:
    #         try:
    #             # saving image using matplotlib
    #             img_4 = np.clip(a=x_hat[0, :, :, :].cpu().numpy().astype(np.float64), a_min=0, a_max=1)
    #         except Exception as e:
    #             logging.info(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
    #             img_4 = np.clip(a=x_hat.sample[0, :, :, :].cpu().numpy().astype(np.float64), a_min=0, a_max=1)
    #         finally:
    #             logging.info(
    #                 f"RGB Clipped Image: x_hat has mean {np.mean(img_4)} and std {np.std(img_4)} and min {np.min(img_4)} and max {np.max(img_4)}. ")
    #             img_4 = np.transpose(img_4, (1, 2, 0))
    #             # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #             fig = plt.figure(dpi=dpi)  # dpi=300
    #             plt.imshow(img_4)
    #             plt.axis("off")
    #             add_custom_figure(writer, step, f"{conditional_txt}CLIPPED_RGB_SAMPLE", fig, model_type="ldm",
    #                               text_condition_raw=text_condition_raw)
    #             plt.close()
    #     except Exception as e:
    #         logging.error(f"Failure while trying to save RGB sample: {e}")
    #
    #     # max value division normalised image
    #     try:
    #         img_3 = x_hat[0, 0, :, :].cpu().numpy().astype(np.float64)
    #     except Exception as e:
    #         logging.info(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
    #         img_3 = x_hat.sample[0, 0, :, :].cpu().numpy().astype(np.float64)
    #     finally:
    #         img3_max = np.max(img_3)
    #         img_3 = img_3 / img3_max
    #
    #         logging.info(
    #             f"max value division:x_hat has mean {np.mean(img_3)} and std {np.std(img_3)} and min {np.min(img_3)} and max {np.max(img_3)}. ")
    #         # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #         fig = plt.figure(dpi=dpi)  # dpi=300
    #         plt.imshow(img_3)
    #         plt.axis("off")
    #         add_custom_figure(writer, step, f"{conditional_txt}MAX_DIV_NORM_GRAYSCALE_SAMPLE", fig, model_type="ldm",
    #                           text_condition_raw=text_condition_raw)
    #         plt.close()
    #
    #     # MinMax-Normalised Grayscale Matplotlib Image
    #     try:
    #         # saving image using matplotlib
    #         img_1 = x_hat[0, 0, :, :].cpu().numpy().astype(np.float64)
    #     except Exception as e:
    #         logging.info(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
    #         img_1 = x_hat.sample[0, 0, :, :].cpu().numpy().astype(np.float64)
    #     finally:
    #         logging.info(
    #             f"Min max norm: x_hat has mean {np.mean(img_1)} and std {np.std(img_1)} and min {np.min(img_1)} and max {np.max(img_1)}. ")
    #         # min max normalisation
    #         img_1 = (img_1 - np.min(img_1)) / (np.max(img_1) - np.min(img_1))
    #         # We want a dpi of 100 (default) rather than an oversized version (dpi = 300)
    #         fig = plt.figure(dpi=dpi)  # dpi=300
    #         plt.imshow(img_1, cmap="gray")
    #         plt.axis("off")
    #         add_custom_figure(writer, step, f"{conditional_txt}MinMax_grayscale_SAMPLE", fig, model_type="ldm",
    #                           text_condition_raw=text_condition_raw)
    #         plt.close()
    #
    #     # Grayscale PIL Image normalised between 0 - 255
    #     try:
    #         # saving image using PIL
    #         from PIL import Image
    #         image = x_hat.sample.cpu().numpy()
    #         image = image.squeeze(0)[0]
    #         image = (image + 1.0) / 2.0
    #         image = image * 255.0
    #         image = image.astype(np.uint8)
    #         image = Image.fromarray(image)
    #         now = str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace(" ", "_")
    #         run_dir = writer.logdir
    #         name = f'Grayscale_PIL_SAMPLE_ldm_step{step}_{now}.png'
    #         image.save(str(run_dir + "/" + name))
    #     except Exception as e:
    #         logging.error(f"Error while trying to save image using PIL: {e}")


def setup_logger(filename='logs.log'):

    # logger to log all stdout to file
    logging.basicConfig(filename=filename,
                       filemode='a',
                       format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S',
                       level=logging.DEBUG)

    # logger to capture and log all stdout messages
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

def add_custom_figure(fig, writer='', step=None, title=None, model_type=None, text_condition_raw=None, epoch=None, enforced_name=None):
    if enforced_name is not None:
        fig.savefig(str(enforced_name))
    else:
        now = str(datetime.datetime.now()).replace(":", "_").replace(".", "_").replace(" ", "_")
        run_dir = writer.logdir
        if epoch is not None:
            name = f'{model_type}_{title}_epoch{epoch}_step{step}_{now}.png'
        else:
            name = f'{model_type}_{title}_step{step}_{now}.png'
        logging.info(f"Now saving figure {name} to {run_dir}.")
        file_name = f"{run_dir}/{name}"
        fig.savefig(file_name, bbox_inches='tight')
        #writer.add_figure(title, fig, step)
        if text_condition_raw != None:
            # Save text condition to file
            with open(str(file_name.replace(".png", ".txt")), "w") as f:
                f.write(text_condition_raw)
        #writer.add_figure(title, fig, step)

def log_controlnet_samples(
        cond: torch.Tensor,
        target: torch.Tensor,
        controlnet: nn.Module,
        diffusion: nn.Module,
        stage1: nn.Module,
        text_encoder: nn.Module,
        scheduler: nn.Module,
        spatial_shape: Tuple,
        writer: SummaryWriter,
        step: int,
        device: torch.device,
        scale_factor: float = 1.0,
        acquisition_time: torch.Tensor = None,
        text_condition: torch.Tensor = None,
        text_condition_raw: str = None,  # original string value of text condition
        dpi: int = 300,
        controlnet_conditioning_scale: float = 1.0,
        dataset_name: str=None,
        epoch:str = None,
    ) -> None:
    """Log a sample from the controlnet-conditioned ldm model decoded by AEKL."""

    conditional_txt = "text-conditioned_" if text_condition_raw is not None else ''

    # Watch out - it is assumed that the noise distribution in the LDM training is distributed similarly as torch.rand((1,) here.
    # -> uniform distribution on the interval [0, 1) -> mean=0.5, std=1/12
    # Watch out - it is assumed that the noise distribution in the LDM training is distributed similarly as torch.rand((1,) here.
    # -> uniform distribution on the interval [0, 1) -> mean=0.5, std=1/12
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    if text_condition == None:
        prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long().to(device)
    else:
        prompt_embeds = text_condition  # get only first report element from batch
    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_embeds.squeeze(1))
    prompt_embeds = prompt_embeds[0]

    if scheduler.timesteps is None:
        scheduler_timesteps = scheduler._timesteps
    else:
        scheduler_timesteps = scheduler.timesteps

    for t in tqdm(scheduler_timesteps, ncols=70):
        with torch.no_grad():
            timestep = torch.asarray((t,)).to(device)
            down_block_res_samples, mid_block_res_sample = sample_from_controlnet(
                controlnet=controlnet,
                noisy_e=latent,
                controlnet_cond=cond,
                timesteps=timestep,
                prompt_embeds=prompt_embeds,
                class_labels=acquisition_time,
                conditioning_scale=controlnet_conditioning_scale,
                )

            noise_pred = sample_from_ldm(model=diffusion,
                                         noisy_e=latent,
                                         timesteps=timestep,
                                         prompt_embeds=prompt_embeds,
                                         down_block_res_samples = down_block_res_samples,
                                         mid_block_res_sample = mid_block_res_sample,
                                         class_labels = acquisition_time
                                         )
            try:
                latent, _ = scheduler.step(noise_pred, t, latent)
            except:
                latent = scheduler.step(noise_pred, t, latent)
            # latent = scheduler.add_noise(original_samples=noise_pred, noise=latent, timesteps=timestep)

    # Let's print some statistics of latent transformed by ldm before re-scaling and decoding.
    logging.info(
        f"Step_{step}: LDMs controlnet-conditioned transformed latent z (shape={latent.shape}, after {len(scheduler_timesteps)} timesteps) "
        f"has mean {torch.mean(latent)} and std {torch.std(latent)} and min {torch.min(latent)} and max {torch.max(latent)}. "
        f"before dividing it by scale_factor ({scale_factor}).")
    with torch.no_grad():
        x_hat = stage1.model.decode(latent / scale_factor)

    try:
        x_hat_numpy = x_hat[0, :, :, :].cpu().numpy()
    except Exception as e:
        x_hat_numpy = x_hat.sample[0, :, :, :].cpu().numpy()

    logging.info(
        f"Step_{step}: x_hat_numpy (after scale factor {scale_factor} multiplication and decoding) "
        f"has mean {np.mean(x_hat_numpy)} and std {np.std(x_hat_numpy)} and min {np.min(x_hat_numpy)} and max {np.max(x_hat_numpy)}. ")

    # TODO: This is the desired output image for 2d to 3d reconstruction of MRI slices (for better intensity correspondence of 2d slices)
    ############### Interval Mapped Grayscale -1,1 RGB Matplotlib Image ###############
    postprocess_and_save_image_triplet(x_hat, target, cond, dpi, writer, step, conditional_txt, dataset_name,text_condition_raw, epoch, acquisition_time)


def sample_from_controlnet(controlnet, noisy_e, controlnet_cond, timesteps, prompt_embeds, class_labels=None,  conditioning_scale=1.0,):
    #logging.debug(f"Shape of class_labels tensor 1: {class_labels.shape}")
    if class_labels is not None and len(class_labels.shape) == 2 and class_labels.shape[0] == 1 and class_labels.shape[1] != 1: # class labels does not have the batch dimension set correctly
        class_labels = torch.squeeze(class_labels, dim=0) # shape from = (1, bs) to (bs)
        #logging.debug(f"Shape of class_labels tensor 2: {class_labels.shape}")
    if class_labels is not None and len(class_labels.shape) == 1:
        class_labels = torch.unsqueeze(class_labels, dim=1) # shape from (bs) to (bs, 1)
        #logging.debug(f"Shape of class_labels tensor 3: {class_labels.shape}")
        #logging.debug(f"Now sampling from controlnet with conditioning_scale={conditioning_scale}. Shape of class_labels tensor: {class_labels.shape}, shape of prompt_embeds: {prompt_embeds.shape}. Shape of noisy_e: {noisy_e.shape}. Shape of timestep: {timesteps.shape}.  ")
    if hasattr(controlnet, "x") or not "ControlNetModel" in str(controlnet.__class__.__name__):
        down_block_res_samples, mid_block_res_sample = controlnet(
            x=noisy_e,
            timesteps=timesteps,
            context=prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale= conditioning_scale,
            class_labels=class_labels
        )
    else:
        down_block_res_samples, mid_block_res_sample = controlnet(
            sample=noisy_e,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            class_labels=class_labels,
            return_dict=False, # get tuple directly
        )
    return down_block_res_samples, mid_block_res_sample



def postprocess_and_save_image_triplet(x_hat, target, cond, dpi=100, writer='', step='', conditional_txt ='', dataset_name='', text_condition_raw='', epoch='', acquisition_time=None, enforced_name=None):
    ############### Interval Mapped Grayscale -1,1 RGB Matplotlib Image ###############
    try:
        try:
            # saving image using matplotlib
            img_1 = x_hat[0].cpu().numpy().astype(np.float64)
        except Exception as e:
            logging.info(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
            img_1 = x_hat.sample[0].cpu().numpy().astype(np.float64)
        img_1 = interval_mapping(img_1, from_min=-1., from_max=1., to_min=0., to_max=1.)
        img_1 = img_1.squeeze(0)  # removing batch dimension
        try:
            img_1 = np.transpose(img_1, (2, 1, 0)).astype(np.float64)  # or (2, 1, 0)
        except Exception as e:
            logging.warning(f"Could not transpose(2, 1, 0) grayscale unchanged image 'img_1' of shape {img_1.shape}: {e}")
        try:
            img_1 = rgb2gray(img_1)  # convert to grayscale (this function also removes channel dim)
        except Exception as e:
            logging.warning(f"Could not convert RGB to grayscale image of shape {img_1.shape}: {e}")

        # assumption here: target is a 3-channel image that still needs to be converted to grayscale
        try:
            target_0 = target.cpu().numpy().astype(np.float64)
            target_0 = interval_mapping(target_0, from_min=-1., from_max=1., to_min=0., to_max=1.)
            target_0 = np.transpose(target_0, (2, 1, 0)).astype(np.float64)  # or (2, 1, 0)
            target_0 = target_0.squeeze(2)
            # target_0 = rgb2gray(target_0)  # convert to grayscale (removing channel dim)
        except Exception as e:
            logging.warning(
                f"Could not convert target to grayscale image of shape H,W,C=1. Shape: {target_0.shape}: {e}")
        # logging.info(f"img_1.shape: {img_1.shape}, target_0.shape: {target_0.shape}.")

        # assumption here: image cond_0 is already grayscale with only a single channel (no need to convert to grayscale)
        cond_0 = cond.cpu().numpy().astype(np.float64)
        try:
            cond_0 = np.transpose(cond_0, (2, 1, 0)).astype(np.float64)  # or (1, 2, 0)
        except Exception as e:
            logging.warning(
                f"Could not transpose(2, 1, 0) cond_0 grayscale unchanged image 'cond_0' of shape cond:{cond_0.shape}: {e}")

        cond_0 = interval_mapping(cond_0, from_min=-1., from_max=1., to_min=0., to_max=1.)
        cond_0 = cond_0.squeeze(2)  # remove channel dimension to make it e.g. (512, 512) instead of (512, 512, 1)
        if cond_0.shape != img_1.shape or target_0.shape != img_1.shape:
            logging.warning(
                f"cond_0.shape: {cond_0.shape} != img_1.shape: {img_1.shape} or != target_0.shape: {target_0.shape}.")
        # logging.info(f"cond_0.shape: {cond_0.shape} != img_1.shape: {img_1.shape} or != target_0.shape: {target_0.shape}.")

        # adding the condition image to the output image to visualize them next to each other
        image = np.concatenate((cond_0, img_1, target_0), axis=1)

        logging.info(
            f"Interval mapped generated image (from -1,1 to 0-1) has mean {np.mean(img_1)} and std {np.std(img_1)} and min {np.min(img_1)} and max {np.max(img_1)}. ")

        logging.info(
            f"Interval mapped cond image (from -1,1 to 0-1) has mean {np.mean(cond_0)} and std {np.std(cond_0)} and min {np.min(cond_0)} and max {np.max(cond_0)}. ")

        logging.info(
            f"Interval mapped concatenated image (from -1,1 to 0-1) has mean {np.mean(image)} and std {np.std(image)} and min {np.min(image)} and max {np.max(image)}. ")

        fig = plt.figure(dpi=dpi)  # dpi=300
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        if enforced_name is not None:
            add_custom_figure(fig=fig, enforced_name=enforced_name)
        else:
            add_custom_figure(fig=fig, writer=writer, step=step, title=f"{conditional_txt}INTERVAL_MAPPED_FROMm11_TO01_GRAYSCALE_SAMPLE",
                          model_type=f"controlnet-{dataset_name}",
                          text_condition_raw=f"{text_condition_raw} | (acquisition_time: {acquisition_time.item() if acquisition_time is not None else 'None'})",
                          epoch=epoch, enforced_name=enforced_name)
        plt.close()
    except Exception as e:
        logging.error(f"Failure while trying to save sample: {e}")





def sample_from_ldm(model, noisy_e, timesteps, prompt_embeds, down_block_res_samples=None, mid_block_res_sample=None,  class_labels=None):
    logging.debug(f"Shape of class_labels tensor 1: {class_labels.shape if class_labels is not None else 'None'}.")
    if class_labels is not None and class_labels.shape[0] == 1 and class_labels.shape[0] != noisy_e.shape[0]: # class labels does not have the batch dimension set correctly
        class_labels = torch.squeeze(class_labels, dim=0) # shape from = (1, bs) to (bs)
        logging.debug(f"Shape of class_labels tensor 2: {class_labels.shape if class_labels is not None else 'None'}.")
        class_labels = torch.unsqueeze(class_labels, dim=1) # shape from (bs) to (bs, 1)
        logging.debug(f"Shape of class_labels tensor 3: {class_labels.shape if class_labels is not None else 'None'}.")
        logging.debug(f"Now sampling from ldm. Shape of class_labels: {class_labels.shape}, shape of prompt_embeds: {prompt_embeds.shape}. Shape of noisy_e: {noisy_e.shape}. Shape of timestep: {timesteps.shape}.")# Shape of down_block_res_samples: {down_block_res_samples.shape if down_block_res_samples is not None else 'None'}. Shape of mid_block_res_sample: {mid_block_res_sample.shape if mid_block_res_sample is not None else 'None'}. ")
    if hasattr(model, "x") or "DiffusionModelUNet" in str(model.__class__.__name__):
        # if diffusion model comes from monai Generative models (i.e. not pretrained)
        if down_block_res_samples is None or mid_block_res_sample is None:
            noise_pred = model(
                x=noisy_e,
                timesteps=timesteps,
                context=prompt_embeds,
                class_labels=class_labels,
                # Note: classifier-free guidance: class_labels will be summed with the timestep embeddings to condition the model output.
            )
        else:
            noise_pred = model(
                x=noisy_e,
                timesteps=timesteps,
                context=prompt_embeds,
                class_labels=class_labels, # Note: classifier-free guidance: class_labels will be summed with the timestep embeddings to condition the model output.
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
    else:
        # if diffusion model comes from diffusers library (i.e. pretrained)
        # in this case we likely need the attributes of forward function of unet2dconditional
        # See: https://huggingface.co/docs/diffusers/api/models/unet2d-cond#diffusers.UNet2DConditionModel.forward
        if down_block_res_samples is None or mid_block_res_sample is None:
            noise_pred = model(
                sample=noisy_e,  # sample instead of x
                timestep=timesteps,  # timestep instead of timestep
                encoder_hidden_states=prompt_embeds,
                # encoder_hidden_states (instead of 'context') seems reserved for prompt embeddings: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L997
                class_labels=class_labels,
                # Note: classifier-free guidance: class_labels will be summed with the timestep embeddings to condition the model output.
            ).sample
        else:
            noise_pred = model(
                sample=noisy_e,  # sample instead of x
                timestep=timesteps,  # timestep instead of timestep
                encoder_hidden_states=prompt_embeds, # encoder_hidden_states (instead of 'context') seems reserved for prompt embeddings: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L997
                class_labels=class_labels, # Note: classifier-free guidance: class_labels will be summed with the timestep embeddings to condition the model output.
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
    return noise_pred

def adjust_class_embedding(model):
    # To condition on continuous acq times, we re-define the class embedding, but as linear projection instead of torch.embedding layer.
    # monai.generative: https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/diffusion_model_unet.py#L2041
    # diffusers: https://github.com/huggingface/diffusers/blob/v0.25.0/src/diffusers/models/unet_2d_condition.py#L349

    # if diffusion model has key num_channels
    if (hasattr(model, "x") or ("DiffusionModelUNet" in str(model.__class__.__name__) or "ControlNet" in str(model.__class__.__name__))) and hasattr(model, 'block_out_channels'):
        # if diffusion model comes from monai Generative models. It has monai.generative parameter naming conventions
        # Setting num_class_embeds required as used for conditioning in forward function of diffusion_model_unet.py
        # See here: https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/diffusion_model_unet.py#L1897
        model.num_class_embeds = int(99999)
        # https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/diffusion_model_unet.py#L2033
        # https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/networks/nets/controlnet.py#L211
        block_out_channels = model.block_out_channels
        time_embed_dim = block_out_channels[0] * 4
        logging.info(f"Setting time_embed_dim to {time_embed_dim}. "
                     f"model.config.time_embedding_dim={time_embed_dim}."
                     f"block_out_channels[0] * 4={block_out_channels[0] * 4}."
                     f"block_out_channels={block_out_channels}.")
    elif hasattr(model, 'config') or "ControlNetModel" in str(model.__class__.__name__):
        model.config.num_class_embeds = int(99999)
        # diffusers naming convention
        block_out_channels = model.config.block_out_channels
        # https://github.com/huggingface/diffusers/blv0.25.0/src/diffusers/models/unet_2d_condition.py#L296
        # FIXME: This only works for positional encoding of time_embed dim. For other embeddings, we need to change the code here.
        time_embed_dim = model.config.time_embedding_dim or block_out_channels[0] * 4
        logging.info(f"Setting time_embed_dim to {time_embed_dim}. "
                     f"model.config.time_embedding_dim={model.config.time_embedding_dim}."
                     f"block_out_channels[0] * 4={block_out_channels[0] * 4}."
                     f"block_out_channels={block_out_channels}.")
    else:
        raise ValueError(
            f"{model.__class__.__name__} has neither key ('x' AND 'block_out_channels') nor key 'config'. Current diffusion model keys: {model.__dict__.keys()}")

    model.class_embedding = nn.Sequential(
        nn.Linear(in_features=1, out_features=time_embed_dim),
        nn.SiLU(),
        nn.Linear(in_features=time_embed_dim, out_features=time_embed_dim),
    )
    #model.class_embedding = nn.Sequential(
    #    nn.Linear(block_out_channels, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
    #)

def compute_weight_tensor(segmentation_mask_batch, weight: int=100):
    # Multiply the segmented area in each segmentation masks in the batch by the weight
    # This is to give more weight to the segmented area in the loss function
    segmentation_mask_batch = segmentation_mask_batch * weight

    # Add +1 to the segmentation mask where the segmentation mask has a value of 0
    # This is to make sure that the background is not ignored (!=0) in the loss function
    segmentation_mask_batch[segmentation_mask_batch == 0] = 1
    return segmentation_mask_batch

def weighted_loss(input_tensor, target_tensor, weight_tensor, loss_function='MSE'):
    if loss_function == 'MAE':
        # Function to measure the element-wise mean absolute error weighted by a weight tensor.
        # MAE error as in https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
        out = torch.abs(input_tensor - target_tensor)
    elif loss_function == 'MSE':
        # Function to measure the element-wise mean squared error weighted by a weight tensor.
        # MSE error as in https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        out = (input_tensor - target_tensor) ** 2
    else:
        raise ValueError(f"loss_function {loss_function} not supported. Use 'MAE' or 'MSE'.")
    out = out * weight_tensor # weight tensor needs to be passed for each sample in batch. Shape: (batch_size, 1 OR 3, height, width)
    loss = out.sum(0) # or sum over whatever dimensions
    return loss

def interval_mapping(image, from_min, from_max, to_min, to_max, is_clipped=True):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    # scale to interval [0,1]
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    # multiply by range and add minimum to get interval [min,range+min]
    x = to_min + (scaled * to_range)
    if is_clipped:
        x = np.clip(x, to_min, to_max)
    return x


def clip_grad(model, clip_grad_at:float=5.0, norm_or_grad: str= 'norm', step='', losses=[], epoch='', verbose=False):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm()
    if clip_grad_at is not None:
        if norm_or_grad == 'norm':
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_at)
        elif norm_or_grad == 'value':
            # https://www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_at)
        else:
            raise ValueError(f"clip_grad_at {norm_or_grad} not supported. Use 'norm' or 'value'.")
    elif norm > 0.9:
        if verbose:
            logging.info(f"Epoch:{epoch}, step:{step}. Exploding gradients of {model.__class__.__name__}? Grad norm is not clipped and is: {norm}")

    grads_after_clipping = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm_after = torch.cat(grads_after_clipping).norm()
    if clip_grad_at is not None and norm > clip_grad_at:
        if verbose or (not verbose and step % 100 == 0):
            logging.info(
                f"Epoch:{epoch}, step:{step}. Clipping grad {norm_or_grad} (initial grad norm: {norm}, grad norm after clip: {norm_after}) at {clip_grad_at} "
                f"for {model.__class__.__name__}. Loss: {losses['loss']}.")
    return norm.item(), norm_after.item()


def get_models_for_controlnet(args, config, scheduler=None, stage1=None, diffusion=None, device=None):
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
        scheduler = PNDMScheduler.from_pretrained(args.source_model, subfolder="scheduler") if scheduler is not None else scheduler# https://arxiv.org/abs/2202.09778
        if hasattr(scheduler, 'prk_timesteps') and scheduler.prk_timesteps is None:
            # To avoid bug when invoking len on NoneType scheduler.prk_timesteps in line 256 of diffusers/schedulers/scheduling_pndm.py,
            # we set the prk_timesteps to an empty array. prk steps are documented to be not useful for stable diffusion in
            # https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/schedulers/scheduling_pndm.py#L204-L206
            #import numpy as np
            #scheduler.prk_timesteps = np.array([]) #
            #scheduler.config.skip_prk_steps = True
            logging.debug(f"Now using {DDPMScheduler.__class__.__name__} from ldm config rather "
                         f"than {scheduler.__class__.__name__} from {args.source_model}. "
                         f"Please change config or code if this is not desired.")
            scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict())) if scheduler is not None else scheduler
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
            diffusion_models = [UNet2DConditionModel.from_pretrained(args.source_model, subfolder="unet"), DiffusionModelUNet(**config["ldm"].get("params", dict()))]
            diffusion = None
            for diffusion_model in diffusion_models:
                if args.cond_on_acq_times:
                    adjust_class_embedding(model=diffusion_model)
                try:
                    ckpt = torch.load(args.ddpm_uri, map_location=torch.device(device))
                    logging.debug(f"Loaded ldm checkpoint file from {args.ddpm_uri}. Contents: {ckpt.keys()}")
                    diffusion_model.load_state_dict(ckpt["diffusion"])
                    logging.info(
                        f"Successfully loaded diffusion_model from ckpt['diffusion'] within the .pt file in {args.ddpm_uri}.")
                except Exception as e1:
                    logging.info(
                        f"Could not load state_dict of the diffusion model using ckpt[diffusion]. Trying to load the state dict from {args.ddpm_uri} directly. Exception: {e1}")
                    try:
                        diffusion_model.load_state_dict(torch.load(args.ddpm_uri, map_location=torch.device(device)))
                        logging.info(
                            f"Successfully loaded diffusion_model directly as state_dict from .pt file in {args.ddpm_uri}.")

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
        scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict())) if scheduler is not None else scheduler
        return config, stage1, diffusion, scheduler

def load_controlnet(config, args, diffusion, device):
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
        if args.cond_on_acq_times: adjust_class_embedding(model=controlnet)

    if ".pt" in args.controlnet_uri:
        logging.info(f"args.controlnet_uri ({args.controlnet_uri}) points to a .pt ckpt file. Loading this ckpt now..")
        try:
            ckpt = torch.load(args.controlnet_uri, map_location=torch.device(device))
            logging.debug(f"Loaded controlnet checkpoint file from {args.controlnet_uri}. Contents: {ckpt.keys()}")
            controlnet.load_state_dict(ckpt["controlnet"])
            logging.info(f"Successfully loaded controlnet from ckpt['controlnet'] within the .pt file in {args.controlnet_uri}.")
        except Exception as e1:
            logging.info(torch.load(args.controlnet_uri, map_location=torch.device(device)))
            logging.info(
                f"Could not load state_dict of the controlnet using ckpt[controlnet]. Trying to load the state dict from {args.controlnet_uri} directly. Exception: {e1}")
            try:
                controlnet.load_state_dict(torch.load(args.controlnet_uri, map_location=torch.device(device)))
                logging.info(f"Successfully loaded controlnet directly as state_dict from .pt file in {args.controlnet_uri}.")
            except Exception as e2:
                logging.info(f"Could not load state_dict directly from {args.controlnet_uri} directly. Exception: {e2}")
                raise e2
    else:
        controlnet = mlflow.pytorch.load_model(args.controlnet_uri)
        logging.info(f"Successfully loaded controlnet using mlflow from .pt file in {args.controlnet_uri}.")
    return controlnet

def get_scale_factor(stage1):
    # Define the vae scaling factor that will be used to (a) multiply vae-encoded latents and (b) divide latents before vae-decoding
    # scaling factor in SD is derived from the length of block_out_channels of vae
    # See code here: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L247C15-L247C83
    # See vae block_out_channels in SD-2-1-base config: https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/vae/config.json#L5
    if stage1.config is not None and stage1.config.block_out_channels is not None:
        len_block_out_channels = len(stage1.config.block_out_channels)
    elif stage1.num_channels is not None:
        len_block_out_channels = len(stage1.num_channels)
    else:
        len_block_out_channels = 4
    scale_factor = 2 ** (len_block_out_channels - 1)
    logging.info(f"Adjusted args.scale_factor for VAE latents to {scale_factor}")
    return scale_factor

def save_controlnet_inference_sample(run_dir, filename, sample, cond, images, dpi=100, save_matplotlib=False, save_comparison=True):

    logging.info(f"Now saving CC-net samples '{filename}' below {run_dir} folder.")

    try:
        # saving image using matplotlib
        sample_postprocessed = sample.sample.cpu().numpy().astype(np.float64)
    except Exception as e:
        logging.warning(f"Error while trying to process stage1-decoded x_hat based on diffusion model's latent: {e}")
        sample_postprocessed = sample.cpu().numpy().astype(np.float64)

    # SAVE USING PIL
    # ONLY SYNTHETIC
    output_file_path = Path(run_dir / "PIL" / filename)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sample_postprocessed = interval_mapping(sample_postprocessed, from_min=-1., from_max=1., to_min=0., to_max=1.)
        #sample_postprocessed = np.clip(sample.cpu().numpy(), min_max[0], min_max[1])
        sample_postprocessed = np.rint((sample_postprocessed * 255)).astype(np.uint8)
        # TODO Here we could use rgb2gray to convert to grayscale instead of using the first channel only
        im = Image.fromarray(sample_postprocessed[0, 0]) # batch and channel dimension removed (?)
        im.save(output_file_path)
        im.close()
        if Path(output_file_path).exists():
            logging.info(f"Saved sample using PIL in {output_file_path}")
    except Exception as e:
        logging.error(f"Failure while trying to save sample using PIL in {output_file_path} : {e}")
    if save_comparison:
        # ONLY SYNTHETIC, SOURCE AND TARGET
        output_file_path = Path(run_dir / "PIL_comparison" / filename)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        #logging.info("cond.shape: " + str(cond.shape),
        #             "images.shape: " + str(images.shape),
        #             "sample_postprocessed.shape: " + str(sample_postprocessed.shape))

        try:
            cond_postprocessed = interval_mapping(cond.cpu().numpy(), from_min=-1., from_max=1., to_min=0., to_max=1.)
            cond_postprocessed = np.rint((cond_postprocessed * 255)).astype(np.uint8)
            images_postprocessed = interval_mapping(images.cpu().numpy(), from_min=-1., from_max=1., to_min=0., to_max=1.)
            images_postprocessed = np.rint((images_postprocessed * 255)).astype(np.uint8)
            sample_postprocessed = np.expand_dims(sample_postprocessed[0, 0, :, :], axis=0)
            cond_postprocessed = np.expand_dims(cond_postprocessed[0, 0, :, :], axis=0)
            images_postprocessed = np.expand_dims(images_postprocessed[0, 0, :, :], axis=0)
            # print shapes
            logging.info(f"sample_postprocessed.shape: {sample_postprocessed.shape}. cond_postprocessed.shape: {cond_postprocessed.shape}. images_postprocessed.shape: {images_postprocessed.shape}")
            #cond_postprocessed = np.concatenate([cond_postprocessed, cond_postprocessed, cond_postprocessed], axis=0)
            sample_triplet = np.concatenate((cond_postprocessed, sample_postprocessed, images_postprocessed), axis=2)
            im = Image.fromarray(sample_triplet[0])
            im.save(output_file_path)
            im.close()
            if Path(output_file_path).exists():
                logging.info(f"Saved sample using PIL_comparison in {output_file_path}")
        except Exception as e:
            logging.error(f"Failure while trying to save sample using PIL_comparison in {output_file_path} : {e}")

    if save_matplotlib:
        # SAVE USING MATPLOTLIB
        # ONLY SYNTHETIC
        output_file_path = Path(run_dir / "MATPLOTLIB" / filename)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            postprocess_and_save_syn_image(x_hat=sample, dpi=dpi, enforced_name=output_file_path)
            if Path(output_file_path).exists():
                logging.info(f"Saved sample using MATPLOTLIB in {output_file_path}")
        except Exception as e:
            logging.error(f"Failure while trying to save sample using postprocess_and_save_syn_image in {output_file_path} : {e}")
        if save_comparison:
            # ONLY SYNTHETIC, SOURCE AND TARGET
            output_file_path = Path(run_dir / "MATPLOTLIB_comparison" / filename)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                postprocess_and_save_image_triplet(x_hat=sample, target=images[0], cond=cond[0], dpi=dpi, enforced_name=output_file_path)
                if Path(output_file_path).exists():
                    logging.info(f"Saved sample using MATPLOTLIB_comparison in {output_file_path}")
            except Exception as e:
                logging.error(f"Failure while trying to save sample using postprocess_and_save_image_triplet in {output_file_path} : {e}")

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
