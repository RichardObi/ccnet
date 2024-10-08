
"""Calculates the radiomics based Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Some code from https://github.com/bioinf-jku/TTUR was reused and adapted i.e. to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import logging
import csv
from pathlib import Path
from typing import List
import SimpleITK as sitk
# from dacite import from_dict
from radiomics.featureextractor import generalinfo, getFeatureClasses, getImageTypes, getParameterValidationFiles, imageoperations
from radiomics import featureextractor

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
# parser.add_argument('--dims', type=int, default=2048,
#                     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
#                     help=('Dimensionality of Inception features to use. '
#                           'By default, uses pool3 features'))
parser.add_argument('--dims', type=int, default=93,
                    help=('Dimensionality of radiomics features to use. '
                          'Depends on which feature types are used. 102 with shape. 93 without.'))
parser.add_argument('--resize_size', type=int, default=None,
                    help=('In case the input images (and masks) are to be resized to a specific pixel dimension. '
                          'Resizing is not yet implemented for 3D data.'))
parser.add_argument('--save-stats', action='store_true',
                    help=('Generate an npz archive from a directory of samples. '
                          'The first path is used as input and the second as output.'))
parser.add_argument('--is_mask_used', action='store_true',
                    help='Generate radiomics based on either available mask or for the whole image.')
parser.add_argument('--mask_dir', type=str, default=None,
                    help='The path to where the masks are located.')


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp', 'nii.gz'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img



def get_activations(files, model, batch_size=1, dims=93, device='cpu',
                    num_workers=1, is_mask_used=True, mask_dir=None, resize_size=None):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    image_paths = []
    mask_paths = []
    radiomics_results = []

    for file_path in files:
        if mask_dir is not None:
            image_paths.append(file_path)
            if not is_mask_used:
                mask_paths.append(None)
            else:
                mask_path1 = Path(mask_dir) / file_path.name.replace("mask", "slice")
                mask_path2 = Path(mask_dir) / file_path.name.replace("slice", "mask")
                mask_path2 = Path(mask_dir) / file_path.name.replace("_synthetic", "")
                mask_path3 = Path(mask_dir) / file_path.name.replace("0001", "0000")
                mask_path4 = Path(mask_dir) / file_path.name.replace("0000", "0001")
                if mask_path1.exists():
                    mask_paths.append(mask_path1)
                elif mask_path2.exists():
                    mask_paths.append(mask_path2)
                elif mask_path3.exists():
                    mask_paths.append(mask_path3)
                elif mask_path4.exists():
                    mask_paths.append(mask_path4)
                else:
                    raise RuntimeError(f"Mask not found for {file_path}. Looked in {mask_path1} and in {mask_path1}")
                    #mask_paths.append(None)
        elif "_mask" in str(file_path) or str(file_path).endswith("_mask.png") or str(file_path).endswith("_mask_synth.jpg"):
            mask_paths.append(file_path)
        else:
            image_paths.append(file_path)
            if not is_mask_used:
                mask_paths.append(None)

    pred_arr = np.empty((len(image_paths), dims))

    for i, (image_path, mask_path) in tqdm(enumerate(zip(image_paths, mask_paths))):
        sitk_image = sitk.ReadImage(str(image_path), outputPixelType=sitk.sitkFloat32)
        if mask_path is None:
            # https://discourse.slicer.org/t/features-extraction/11047/3
            ma_arr = np.ones(sitk_image.GetSize()[::-1])  # reverse the order as image is xyz, array is zyx
            sitk_mask = sitk.GetImageFromArray(ma_arr)
            sitk_mask.CopyInformation(sitk_image)  # Copy geometric info
            #print("No mask is used for feature extraction.")
        else:
            sitk_mask = sitk.ReadImage(str(mask_path))

        # Check if the mask is in range [0, 255] and rescale it to [0, 1]
        if np.max(sitk.GetArrayFromImage(sitk_mask)) == 255:
            sitk_mask = sitk.Cast(sitk_mask, sitk.sitkFloat32) / 255.0

        if i % 20 == 0:
            # get some prints to check the progress and if everything is working
            print("Processing image:", image_path)
            print("Processing mask:", mask_path)

        if resize_size is not None:
            sitk_image_array = sitk.GetArrayFromImage(sitk_image)
            sitk_image_array_resized = cv2.resize(sitk_image_array, (resize_size,resize_size), interpolation=cv2.INTER_LINEAR)
            sitk_image_resized = sitk.GetImageFromArray(sitk_image_array_resized)
            try:
                sitk_image_resized.CopyInformation(sitk_image)
            except:
                pass
            sitk_image = sitk_image_resized
            sitk_mask_array = sitk.GetArrayFromImage(sitk_mask)
            sitk_mask_array_resized = cv2.resize(sitk_mask_array, (resize_size,resize_size), interpolation=cv2.INTER_LINEAR)
            # After resizing, set all values above 0.5 to 1 and all values below to 0
            sitk_mask_array_resized[sitk_mask_array_resized > 0.5] = 1
            sitk_mask_array_resized[sitk_mask_array_resized <= 0.5] = 0
            sitk_mask_resized = sitk.GetImageFromArray(sitk_mask_array_resized)
            try:
                sitk_mask_resized.CopyInformation(sitk_mask)
            except:
                pass
            sitk_mask = sitk_mask_resized


        # Check if the mask contains only one voxel. This needs to be done before and after resizing as the mask
        if np.sum(sitk.GetArrayFromImage(sitk_mask)) <= 1:
            print("Skipping mask with only one segmented voxel:", mask_path)
            continue

        # Finally, run the feature extraction
        try:
            output = model.execute(sitk_image, sitk_mask)
        except Exception as e:
            print(f"Error occurred while extracting features for image {i} from image {image_path} and mask {mask_path}: {e}")
            raise e
        radiomics_features = {}
        for feature_name in output.keys():
            if "diagnostics" not in feature_name:
                radiomics_features[feature_name.replace("original_", "")] = float(output[feature_name])

        radiomics_results.append(radiomics_features)

        pred_arr[i] = list(radiomics_features.values())
        print(f"Total number of features extracted for image {i}: {len(pred_arr[i])}")

    if radiomics_results:
        sample_dict = radiomics_results[0]
        num_features = len(sample_dict)
        print("Number of radiomics features:", num_features)

    return pred_arr, radiomics_results, image_paths, mask_paths

def save_features_to_csv(csv_file_path, image_paths, mask_paths, feature_data):
    """Save the feature data to a CSV file.

    Params:
    -- csv_file_path   : Path to the CSV file where the results will be saved
    -- image_paths     : List of image file paths
    -- mask_paths      : List of mask file paths
    -- feature_data    : Feature data to be saved in the CSV file
    """
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        header = ["image_path", "mask_path"]
        for feature_name in feature_data[0].keys():
            header.append(feature_name)
        writer.writerow(header)

        # Write the rows for each image
        for image_path, mask_path, features in zip(image_paths, mask_paths, feature_data):
            if mask_path is not None:
                mask_path = mask_path.with_name(mask_path.name.replace("_img_synth.jpg", "_mask_synth.jpg"))
            row = [str(image_path), str(mask_path)]
            row.extend(features.values())
            writer.writerow(row)

        # Compute and save the min and max values for each column
        num_features = len(feature_data[0])
        min_values = [np.min([data[feature_name] for data in feature_data]) for feature_name in feature_data[0].keys()]
        max_values = [np.max([data[feature_name] for data in feature_data]) for feature_name in feature_data[0].keys()]
        empty_row = [''] * (num_features + 2)  # Create an empty row to separate the data

        # Write the rows for min values
        writer.writerow(empty_row)
        writer.writerow(['Min', ''] + min_values)

        # Write the rows for max values
        writer.writerow(empty_row)
        writer.writerow(['Max', ''] + max_values)

    print("Feature data saved to", csv_file_path)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def min_max_normalize(features, new_min, new_max):
    # Calculate the minimum and maximum values of each feature across all images
    min_values = np.nanmin(features, axis=0)
    max_values = np.nanmax(features, axis=0)

    # Create a new copy of features to perform normalization
    normalized_features = np.copy(features)

    # Perform Min-Max normalization for columns with different min and max values
    for idx, (min_val, max_val) in enumerate(zip(min_values, max_values)):
        if not np.isnan(min_val) and not np.isnan(max_val):
            normalized_features[:, idx] = ((features[:, idx] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

    # Replace NaN values with the mean of new_min and new_max
    nan_indices = np.isnan(normalized_features)
    mean_value = (new_min + new_max) / 2
    normalized_features[nan_indices] = mean_value

    return normalized_features

def calculate_activation_statistics(files, model, batch_size=1, dims=93,
                                    device='cpu', num_workers=1, is_mask_used=True, mask_dir=None, resize_size=None):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act, radiomics_results, image_paths, mask_paths = get_activations(files, model, batch_size, dims, device, num_workers, is_mask_used=is_mask_used, mask_dir=mask_dir, resize_size=resize_size)
    print('features of radiomics------', act)
    # print('features of radiomics shape------', type(act))

    # Extract the folder name from the first image file path
    folder_name = Path(files[0]).parent.stem

    # Generate a unique identifier using the current timestamp
    unique_identifier = int(time.time())

    # Define the CSV file path with a unique identifier and the folder name in the name
    csv_file_path = f"radiomics_results_{folder_name}_{unique_identifier}.csv"

    save_features_to_csv(csv_file_path, image_paths, mask_paths, radiomics_results)

    # to check NaN values in features
    features = act

    if np.isnan(features).any():
        nan_indices = np.where(np.isnan(features))
        unique_nan_indices = np.unique(nan_indices[1])
        print("Warning: NaN values detected in the features array.")
        print("Number of NaN values for each feature:")
        for feature_idx in unique_nan_indices:
            nan_count = np.sum(np.isnan(features[:, feature_idx]))
            print(f"Feature {feature_idx}: {nan_count} NaN values")

            # Get the row indices with NaN values for this feature
            row_indices_with_nan = nan_indices[0][nan_indices[1] == feature_idx]

            print(f"Row indices with NaN values for Feature {feature_idx}: {row_indices_with_nan}")


    normalized_act = min_max_normalize(act, 0, 7.45670747756958)
    print('features found are as follows', act)
    print('normalized_features----------',normalized_act)
    
    norm_csv_file_path = f"radiomics_results_normalized_{folder_name}_{unique_identifier}.csv"
    # save_features_to_csv(csv_file_path, image_paths, mask_paths, radiomics_results)
   
    # for normalized features:
    # Write the data to the CSV file
    with open(norm_csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Iterate through the rows and write each element to a new row in the CSV file
        for row in normalized_act:
            writer.writerow(row)

    mu = np.mean(normalized_act, axis=0)
    sigma = np.cov(normalized_act, rowvar=False)
    # print('mu and sigma-----------------------------++++++++++++++++', mu, sigma)
    # print('mu and sigma length_____________________', len(mu), len(sigma))
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1, is_mask_used=True, mask_dir=None, resize_size=None):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        # print('files in compute_statistics_of_path', files)
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers, is_mask_used=is_mask_used, mask_dir=mask_dir, resize_size=resize_size)
    # print('m and s********************', m, s)
    return m, s



def get_feature_extractor(features_to_compute = None, settings:dict = None):
    """Returns a pyradiomics feature extractor"""
    # Customize the list of radiomics features to compute based on your dataset
    # For 2D datasets like BCDR Dataset, use "shape2D"
    # For 3D datasets like Duke Dataset, use "shape"
    if features_to_compute is None:
        features_to_compute: List[str] = [
            "firstorder",
            #"shape2D",  # Modify this line as needed
            #"shape",  # Modify this line as needed
            "glcm",
            "glrlm",
            "gldm",
            "glszm",
            "ngtdm",
        ]
    if settings is None:
        settings = {}
    radiomics_results = []
    # Resize mask if there is a size mismatch between image and mask
    settings["setting"] = {"correctMask": True}
    # Set the minimum number of dimensions for a ROI mask. Needed to avoid error, as in our datasets we have some masses with dim=1.
    # https://pyradiomics.readthedocs.io/en/latest/radiomics.html#radiomics.imageoperations.checkMask
    settings["setting"] = {"minimumROIDimensions": 1}

    # Set feature classes to compute
    settings["featureClass"] = {feature: [] for feature in features_to_compute}
    return featureextractor.RadiomicsFeatureExtractor(settings)


def calculate_frd_given_paths(paths, batch_size, device, dims, num_workers=1, is_mask_used=True, mask_dir=None, resize_size=None):
    """Calculates the FID of two paths"""

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    model = get_feature_extractor()

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, is_mask_used=is_mask_used, mask_dir=mask_dir, resize_size=resize_size)
    # print('m1, s1------------------------------', m1, s1)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers, is_mask_used=is_mask_used, mask_dir=mask_dir, resize_size=resize_size)
    # print('m2, s2------------------------------', m2, s2)
    frd_value = calculate_frechet_distance(m1, s1, m2, s2)

    return frd_value

def save_frd_stats(paths, batch_size, device, dims, num_workers=1, is_mask_used=True, mask_dir=None, resize_size=None):
    """Calculates the FID of two paths"""
    if not os.path.exists(paths[0]):
        raise RuntimeError('Invalid path: %s' % paths[0])

    if os.path.exists(paths[1]):
        raise RuntimeError('Existing output file: %s' % paths[1])

    model = get_feature_extractor()

    print(f"Saving statistics for {paths[0]}")

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers, is_mask_used=is_mask_used, mask_dir=mask_dir, resize_size=resize_size)

    np.savez_compressed(paths[1], mu=m1, sigma=s1)


def main():
    args = parser.parse_args()
    print(args)

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_frd_stats(args.path, args.batch_size, device, args.dims, num_workers, is_mask_used=args.is_mask_used, mask_dir=args.mask_dir, resize_size=args.resize_size)
        return

    frd_value = calculate_frd_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers,
                                          is_mask_used=args.is_mask_used,
                                          mask_dir=args.mask_dir,
                                          resize_size=args.resize_size,
                                          )
    print('FRD: ', frd_value)


if __name__ == '__main__':
    main()





