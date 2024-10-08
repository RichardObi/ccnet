"""
Feature Extractor

RadImageNet Model source: https://github.com/BMEII-AI/RadImageNet
RadImageNet InceptionV3 weights (updated link 11.07.2023): https://drive.google.com/drive/folders/1lGFiS8_a5y28l4f8zpc7fklwzPJC-gZv

Usage:
    python features_extractor.py dir1 dir2
"""
import argparse
import random
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import wget
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from csv import writer


tf.autograph.set_verbosity(3)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

random.seed(123)
np.random.seed(123)

img_size = 299
num_batches = 1
RADIMAGENET_URL = "https://drive.google.com/uc?id=1uvJHLG1K71Qzl7Km4JMpNOwE7iTjN8g9"
RADIMAGENET_WEIGHTS = ["RadImageNet-InceptionV3_notop.h5"]
IMAGENET_TFHUB_URL = "https://tfhub.dev/tensorflow/tfgan/eval/inception/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature extractor using inception v3 trained on imagenet or radimagenet."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to images from first dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imagenet",
        help="Use RadImageNet feature extractor?",
    )

    parser.add_argument(
        "--normalize_images",
        action="store_true",
        help="Normalize images from both data sources using min and max of each sample",
    )
    args = parser.parse_args()
    return args

def load_images_from_filelist(file_names, normalize=False):
    """
    Loads images from the given directory.
    If split is True, then half of the images is loaded to one array and the other half to another.
    """

    images = []
    for count, filename in enumerate(file_names):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(filename)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            if normalize:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            if len(img.shape) > 2 and img.shape[2] == 4:
                img = img[:, :, :3]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            images.append(img)
        return np.array(images)

def check_model_weights(model_name):
    """
    Checks if the model weights are available and download them if not.
    """
    model_weights_path = None
    if model_name == "radimagenet":
        for radimagenet_weight_path in RADIMAGENET_WEIGHTS:
            if os.path.exists(radimagenet_weight_path):
                model_weights_path = radimagenet_weight_path
                break
        if model_weights_path is None:
            model_weights_path = RADIMAGENET_WEIGHTS[0]
            print(f"Downloading RadImageNet InceptionV3 model to be stored in {model_weights_path}:")
            wget.download(
                RADIMAGENET_URL,
                model_weights_path,
            )
            print("\n")
        return model_weights_path


def _radimagenet_fn(images):
    """
    Get RadImageNet inception v3 model
    """
    model_weights_path = None
    for radimagenet_weight_path in RADIMAGENET_WEIGHTS:
        if os.path.exists(radimagenet_weight_path):
            model_weights_path = radimagenet_weight_path
            break
    model = InceptionV3(
        weights=model_weights_path,
        input_shape=(img_size, img_size, 3),
        include_top=False,
        pooling="avg",
    )
    output = model(images)
    output = tf.nest.map_structure(tf.keras.layers.Flatten(), output)
    return output


def get_classifier_fn(model_name="imagenet"):
    """
    Get model as TF function for optimized inference.
    """
    check_model_weights(model_name)

    if model_name == "radimagenet":
        return _radimagenet_fn
    elif model_name == "imagenet":
        return tfgan.eval.classifier_fn_from_tfhub(IMAGENET_TFHUB_URL, "pool_3", True)
    else:
        raise ValueError("Model {} not recognized".format(model_name))

    # Pass images through the classifier to get the activations/features.
    return get_activations(preprocess_input(images_1), get_classifier_fn(model_name), num_batches=num_batches)


def get_activations(input_tensor1, classifier_fn, num_batches=1):
    """A helper function for evaluating the frechet classifier distance."""

    # Compute the activations using the memory-efficient `map_fn`.
    def compute_activations(elems):
        return tf.map_fn(
            fn=classifier_fn,
            elems=elems,
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier')
    #activations = tf.stack(tf.split(input_tensor1, num_or_size_splits=num_batches))
    #activations = compute_activations(activations)
    #activations = tf.unstack(activations, 0)
    #return tf.concat(activations)
    # Ensure the activations have the right shapes.
    return tf.concat(tf.unstack(compute_activations(tf.stack(tf.split(input_tensor1, num_or_size_splits=num_batches)))), 0)

if __name__ == "__main__":
    args = parse_args()
    print(f"Args for feature extractor: {args}")
    directory = args.dataset_path

    file_names = sorted(glob.glob(f'{directory}/*.png')) if ".png" in os.listdir(directory)[0] else sorted(glob.glob(f'{directory}/*.jpg'))

    images = load_images_from_filelist(file_names, normalize=args.normalize_images)
    image_tensor = preprocess_input(images)
    classifier_fn = get_classifier_fn(args.model_name)
    features = get_activations(image_tensor, classifier_fn, num_batches=args.num_batches)
    #print(f"Extracted features: {features}")

    # Open existing CSV file in append mode and add features
    with open('features.csv', 'a') as f_object:
        writer_object = writer(f_object)
        for idx, feature in enumerate(features):
            writer_object.writerow([file_names[idx], feature])
        f_object.close()



