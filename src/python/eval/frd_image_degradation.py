import os
import shutil
import SimpleITK as sitk
import numpy as np
import cv2
import argparse

# def calculate_kernel_size(original_size, percentage_blur, factor=0.001):
#     """
#     Calculate the kernel size for Gaussian blur based on the percentage of blur.

#     Args:
#         original_size (tuple): The size of the original image in (width, height, depth) format.
#         percentage_blur (int): The percentage of blur to apply.
#         factor (float): A factor to reduce the impact of blur (default is 0.1).

#     Returns:
#         int: The calculated kernel size for Gaussian blur.
#     """
#     max_dimension = max(original_size)
#     kernel_size = int((percentage_blur / 100) * max_dimension * factor)
#     kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
#     return kernel_size

def calculate_kernel_size(original_size, percentage_blur, factor=0.001):
    """
    Calculate the kernel size for Gaussian blur based on the percentage of blur.

    Args:
        original_size (tuple): The size of the original image in (width, height, depth) format.
        percentage_blur (float): The percentage of blur to apply.
        factor (float): A factor to adjust the sensitivity of blur (default is 0.001).

    Returns:
        int: The calculated kernel size for Gaussian blur.
    """
    # Adjust the sensitivity of blur based on the factor
    blur_sensitivity = factor * percentage_blur

    # Calculate the kernel size based on the adjusted blur sensitivity
    max_dimension = max(original_size)
    kernel_size = int((percentage_blur / 100) * max_dimension + blur_sensitivity)

    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
    return kernel_size


def generate_output_folder(input_folder, percentage_blur):
    """
    Generate the name of the output folder for blurred images.

    Args:
        input_folder (str): Path to the input folder containing images.
        percentage_blur (int): The percentage of blur applied to the images.

    Returns:
        str: The name of the output folder.
    """
    base_folder = os.path.basename(input_folder)
    return f"{base_folder}_blur_{percentage_blur}_percent"

def add_blur_to_images(input_folder, percentage_blurs, output_folder):
    """
    Apply Gaussian blur to images in the input folder and save them to the output folder.

    Args:
        input_folder (str): Path to the input folder containing images.
        percentage_blurs (list): List of blur percentages to apply.
        output_folder (str): Path to the output folder where blurred images will be saved.
    """
    for percentage_blur in percentage_blurs:
        print(f"Applying {percentage_blur}% blur...")
        if output_folder is None:
            blur_folder = generate_output_folder(input_folder, percentage_blur)
        else:
            blur_folder = os.path.join(output_folder, generate_output_folder(input_folder, percentage_blur))
        
        if not os.path.exists(blur_folder):
            os.makedirs(blur_folder)

        for filename in os.listdir(input_folder):
            if (filename.endswith(".png") and "_mask" not in filename) or (filename.endswith(".nii.gz") and "_0001" in filename):
                print(f"Processing file: {filename}")
                image_path = os.path.join(input_folder, filename)

                # Load the image using SimpleITK
                print("Loading image...")
                image = sitk.ReadImage(image_path)
                print("Image loaded.")

                # Convert the SimpleITK image to a NumPy array
                image_array = sitk.GetArrayFromImage(image)

                # Normalize the image array
                image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

                # Calculate the kernel size based on the percentage of blur
                kernel_size = calculate_kernel_size(image.GetSize(), percentage_blur)
                print(f"Kernel size for {percentage_blur}% blur: {kernel_size}")

                # Apply Gaussian blur to the numpy array
                print("Applying Gaussian blur...")
                blurred_array = cv2.GaussianBlur(image_array.astype(np.float32), (kernel_size, kernel_size), 0)
                print("Gaussian blur applied.")

                # Scale the pixel values back to the range [0, 1]
                blurred_array = np.clip(blurred_array, 0, 1)

                # Convert the blurred numpy array back to a SimpleITK image
                blurred_image_sitk = sitk.GetImageFromArray((blurred_array * 255).astype(np.uint8))
                blurred_image_sitk.CopyInformation(image)

                # Save the blurred image to the output folder
                output_path = os.path.join(blur_folder, filename)
                print(f"Saving blurred image to {output_path}...")
                sitk.WriteImage(blurred_image_sitk, output_path)
                print("Blurred image saved.")
        
            else:
                # For non-matching images (not ending with .nii.gz)
                # Copy the original image to the output folder
                original_image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(blur_folder, filename)
                print(f"Copying original image to {output_path}...")
                shutil.copy(original_image_path, output_path)
                print("Original image copied.")

def main():
    """
    Main function to parse command-line arguments and initiate image blurring.
    """
    parser = argparse.ArgumentParser(description="Add blur to images in an input folder.")
    parser.add_argument("input_folder", help="Path to the input folder containing images.")
    parser.add_argument("-o", "--output_base_folder", help="Path to the output folder (default is None).")
    parser.add_argument("-p", "--percentages", nargs="+", type=int, default=[0.1, 1, 5, 10, 20, 50], help="List of blur percentages.")
    args = parser.parse_args()

    # If output_folder is not provided, set it to None
    output_folder = args.output_base_folder

    add_blur_to_images(args.input_folder, args.percentages, output_folder)

if __name__ == "__main__":
    main()






















# import os
# import shutil
# import SimpleITK as sitk
# import numpy as np
# import cv2
# import argparse

# def calculate_kernel_size(original_size, percentage_blur, factor=0.1):
#     """
#     Calculate the kernel size for Gaussian blur based on the percentage of blur.

#     Args:
#         original_size (tuple): The size of the original image in (width, height, depth) format.
#         percentage_blur (int): The percentage of blur to apply.
#         factor (float): A factor to reduce the impact of blur (default is 0.1).

#     Returns:
#         int: The calculated kernel size for Gaussian blur.
#     """
#     max_dimension = max(original_size)
#     kernel_size = int((percentage_blur / 100) * max_dimension * factor)
#     kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
#     return kernel_size


# def generate_output_folder(input_folder, percentage_blur):
#     """
#     Generate the name of the output folder for blurred images.

#     Args:
#         input_folder (str): Path to the input folder containing images.
#         percentage_blur (int): The percentage of blur applied to the images.

#     Returns:
#         str: The name of the output folder.
#     """
#     base_folder = os.path.basename(input_folder)
#     return f"{base_folder}_blur_{percentage_blur}_percent"

# def add_blur_to_images(input_folder, percentage_blurs, output_folder):
#     """
#     Apply Gaussian blur to images in the input folder and save them to the output folder.

#     Args:
#         input_folder (str): Path to the input folder containing images.
#         percentage_blurs (list): List of blur percentages to apply.
#         output_folder (str): Path to the output folder where blurred images will be saved.
#     """
#     for percentage_blur in percentage_blurs:
#         print(f"Applying {percentage_blur}% blur...")
#         if output_folder is None:
#             blur_folder = generate_output_folder(input_folder, percentage_blur)
#         else:
#             blur_folder = os.path.join(output_folder, generate_output_folder(input_folder, percentage_blur))
        
#         if not os.path.exists(blur_folder):
#             os.makedirs(blur_folder)

#         for filename in os.listdir(input_folder):
#             if (filename.endswith(".png") and "_mask" not in filename) or (filename.endswith(".nii.gz") and "_0001" in filename):
#                 print(f"Processing file: {filename}")
#                 image_path = os.path.join(input_folder, filename)

#                 # Load the image using SimpleITK
#                 print("Loading image...")
#                 image = sitk.ReadImage(image_path)
#                 print("Image loaded.")

#                 # Calculate the kernel size based on the percentage of blur
#                 kernel_size = calculate_kernel_size(image.GetSize(), percentage_blur)
#                 print(f"Kernel size for {percentage_blur}% blur: {kernel_size}")

#                 # Convert the SimpleITK image to a NumPy array
#                 image_array = sitk.GetArrayFromImage(image)

#                 # Apply Gaussian blur to the numpy array
#                 print("Applying Gaussian blur...")
#                 blurred_array = np.zeros_like(image_array)
#                 for z in range(image.GetDepth()):
#                     blurred_array[z] = cv2.GaussianBlur(image_array[z], (kernel_size, kernel_size), 0)
#                 print("Gaussian blur applied.")

#                 # Convert the blurred numpy array back to a SimpleITK image
#                 blurred_image_sitk = sitk.GetImageFromArray(blurred_array)
#                 blurred_image_sitk.CopyInformation(image)

#                 # Save the blurred image to the output folder
#                 output_path = os.path.join(blur_folder, filename)
#                 print(f"Saving blurred image to {output_path}...")
#                 sitk.WriteImage(blurred_image_sitk, output_path)
#                 print("Blurred image saved.")
        
#             else:
#                 # For non-matching images (not ending with .nii.gz)
#                 # Copy the original image to the output folder
#                 original_image_path = os.path.join(input_folder, filename)
#                 output_path = os.path.join(blur_folder, filename)
#                 print(f"Copying original image to {output_path}...")
#                 shutil.copy(original_image_path, output_path)
#                 print("Original image copied.")

# def main():
#     """
#     Main function to parse command-line arguments and initiate image blurring.
#     """
#     parser = argparse.ArgumentParser(description="Add blur to images in an input folder.")
#     parser.add_argument("input_folder", help="Path to the input folder containing images.")
#     parser.add_argument("-o", "--output_base_folder", help="Path to the output folder (default is None).")
#     parser.add_argument("-p", "--percentages", nargs="+", type=int, default=[0.1], help="List of blur percentages.")
#     args = parser.parse_args()
#     # [0.1, 1, 5, 10, 20, 50
#     # If output_folder is not provided, set it to None
#     output_folder = args.output_base_folder

#     add_blur_to_images(args.input_folder, args.percentages, output_folder)

# if __name__ == "__main__":
#     main()
