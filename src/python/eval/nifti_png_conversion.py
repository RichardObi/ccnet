# Based on: https://ianmcatee.com/converting-a-nifti-file-to-an-image-sequence-using-python/

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
import os
from tqdm import tqdm
import time
import math
# global variables


LOCAL_TEST = False # TODO False for server

IS_SEGMENTATION = True  # Set true if you want to convert the segmentation masks to pngs
SKIP_TRAINING = True

DIGITS_TO_STORE = [0, 1] #[0, 1] #, 2, 3, 4, 5, 6, 7, 8, 9] # the different phases
print(f"WARNING: Digits (DCE Phases) to be stored: {DIGITS_TO_STORE}")

### Server
PREFIX_PATH = os.path.join('/home/roo/data/RadioVal/riti/Radioval_data/', '')
PREFIX_PATH2 = os.path.join('/home/roo/data/RadioVal/richard/Duke-Breast-Cancer-MRI-Nifti-Whole-No-Masks','')   #manifest-1686223866660/', '')
#PREFIX_OUTPUT_PATH = os.path.join('/home/roo/data/RadioVal/richard/ldm_data_phase_one_w_mask', '')  # PREFIX_PATH
#PREFIX_OUTPUT_PATH = os.path.join('/home/roo/Desktop/software/breastmri/data/duke_all_png_slices_1to196_slide_greater196', '')  # PREFIX_PATH
#PREFIX_OUTPUT_PATH = os.path.join('/home/roo/Desktop/software/breastmri/data/duke_maemi_experiment', '')  # PREFIX_PATH
PREFIX_OUTPUT_PATH = os.path.join('/home/roo/Desktop/software/breastmri/data/MPost', '')  # PREFIX_PATH


if LOCAL_TEST:
    ### Local
    PREFIX_PATH = os.path.join('/Users/richardosuala/Desktop/Radioval_data/', '')
    PREFIX_OUTPUT_PATH = os.path.join(PREFIX_PATH, f'Test_Output_{time.strftime("%Y%m%d_%H%M%S")}/')
    PREFIX_PATH2 = PREFIX_PATH

INPUT_FOLDER_PATH = os.path.join(PREFIX_PATH, 'Duke-Breast-Cancer-MRI-Nifti-Whole')
OUTPUT_FOLDER_PATH = os.path.join(PREFIX_OUTPUT_PATH, 'Duke-Breast-Cancer-MRI-png-Whole')

if IS_SEGMENTATION:
    INPUT_FOLDER_PATH = os.path.join(PREFIX_PATH, 'correct_masks') # TODO folder name might be different on server
    #OUTPUT_FOLDER_PATH = os.path.join(PREFIX_OUTPUT_PATH, 'correct_masks-png')


CSV_PATHS = []
CSV_PATHS.append(os.path.join(PREFIX_PATH, 'Duke_Breast_MRI_all_phases.csv'))
CSV_PATHS.append(os.path.join(PREFIX_PATH2, 'Duke_Breast_MRI_all_phases_without_masks.csv')) # TODO Uncomment for server
VERBOSE = False  # True to display the middle slices of the scan and nifti header information
FOR_PIX2PIX = True  # Set true to store in pix2pixHD desired folder structure (train_A, train_B)
USE_BOX_ANNOTATIONS = True  # Set true if you want to use the box annotations to extract a range around the tumor-containing slides from the Duke dataset
EXTRACT_TUMOR_ROI_ONLY = False  # Requires: USE_BOX_ANNOTATIONS==True. If true, then we extract only the tumor-containing slides (also, as before, the range around them) from the Duke dataset cropping a ROI around the tumor and resize
EXTRACT_TUMOR_ROI_SIZE = None #224 # Requires: USE_BOX_ANNOTATIONS==True and EXTRACT_TUMOR_ROI_ONLY==True. Specifies the resize size of the tumor ROI after ROI extraction. # 224x224 is imagenet dim and therefore could be a good value for downstream tasks
BOX_AXIAL_MARGIN_PERCENTAGE = 0.05 if EXTRACT_TUMOR_ROI_ONLY is None else 0.0 # e.g. 0.1 = 10% of volume left and right of the tumor bb  # Extract the percentage of additional slices to be extracted before and after the tumor-containing MRI slices in axial direction from the Duke dataset
BOX_ANNOTATIONS_FILE_PATH = os.path.join(PREFIX_PATH, 'Annotation_Boxes.xlsx')
EXTRACT_SINGLE_BREAST_QUADRATIC_ROI=False # Requires: USE_BOX_ANNOTATIONS==True and EXTRACT_TUMOR_ROI_ONLY==True. If true, then we crop the breast (left or right) before extracting from only the tumor-containing slides. We add a margin around the height of the bounding box to make it quadratic.

EXTRACT_BOUNDING_BOX_MASK=True # Requires: USE_BOX_ANNOTATIONS==True. If true, then we extract the bounding box around the tumor setting it to 255 and all non-bounding box pixel to 0.

RESIZE_TO = 512 if EXTRACT_TUMOR_ROI_SIZE is None else EXTRACT_TUMOR_ROI_SIZE  # None

if EXTRACT_SINGLE_BREAST_QUADRATIC_ROI:
    print(f"EXTRACT_SINGLE_BREAST_QUADRATIC_ROI=={EXTRACT_SINGLE_BREAST_QUADRATIC_ROI}")
    RESIZE_TO = None # No resizing for this case, as we may want to do the resizing in the generative model pipeline (testing different resize scales)


VIEWS = ['axial']  # ['sagital', 'coronal', 'axial']
#STORE_GRAYSCALE = True

# specific approach for data to be used in pix2pixHD
if FOR_PIX2PIX:
    OUTPUT_FOLDER_PATH = [os.path.join(PREFIX_OUTPUT_PATH, 'train', 'train_A'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'train', 'train_B'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'test', 'test_A'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'test', 'test_B'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'validation', 'validation_A'),
                          os.path.join(PREFIX_OUTPUT_PATH, 'validation', 'validation_B')]

    VIEWS = ['axial']
    # Min slide- previously tried with:  #25 #20
    # 1 is the smallest number of slice indices that contains a tumour in the Duke Dataset (see Annotation_Boxes.xlsx)
    # SLIDE_MIN = 0

    # Max slide- previously tried with:  #120 #125
    # 196 is the largest number of slice indices that contains a tumour in the Duke Dataset (see Annotation_Boxes.xlsx)
    # SLIDE_MAX = 196
    SLIDE_MIN = -1
    SLIDE_MAX = 999999



if USE_BOX_ANNOTATIONS:
    # read in the box annotations
    import pandas as pd
    box_annotations = pd.read_excel(BOX_ANNOTATIONS_FILE_PATH, sheet_name='Sheet1')
    # For each case, get the end_slice and start_slice of the box annotation
    box_annotations_dict = {}
    for index, row in box_annotations.iterrows():
        case = row['Patient ID']
        start_slice = row['Start Slice']
        end_slice = row['End Slice']
        if EXTRACT_TUMOR_ROI_ONLY:
            start_row = row['Start Row']
            end_row = row['End Row']
            start_column = row['Start Column']
            end_column = row['End Column']
            box_annotations_dict[case] = [start_slice, end_slice, start_row, end_row, start_column, end_column]
        else:
            box_annotations_dict[case] = [start_slice, end_slice, None, None, None, None]
    #print(box_annotations_dict)


# The below validation and testset lists are for MAEMI from Lang et al
VALIDATIONSET_LIST = ["Breast_MRI_084","Breast_MRI_892","Breast_MRI_697","Breast_MRI_451","Breast_MRI_813","Breast_MRI_865","Breast_MRI_763","Breast_MRI_475","Breast_MRI_802","Breast_MRI_324","Breast_MRI_705","Breast_MRI_271","Breast_MRI_902","Breast_MRI_232","Breast_MRI_294","Breast_MRI_722","Breast_MRI_300","Breast_MRI_604","Breast_MRI_509","Breast_MRI_445","Breast_MRI_834","Breast_MRI_798","Breast_MRI_096","Breast_MRI_150","Breast_MRI_278","Breast_MRI_730","Breast_MRI_573","Breast_MRI_211","Breast_MRI_685","Breast_MRI_349","Breast_MRI_125","Breast_MRI_226","Breast_MRI_136","Breast_MRI_273","Breast_MRI_167","Breast_MRI_019","Breast_MRI_036","Breast_MRI_085","Breast_MRI_833","Breast_MRI_874","Breast_MRI_106","Breast_MRI_569","Breast_MRI_148","Breast_MRI_731","Breast_MRI_593","Breast_MRI_518","Breast_MRI_723","Breast_MRI_210","Breast_MRI_459","Breast_MRI_004"]
TESTSET_LIST = ["Breast_MRI_365","Breast_MRI_263","Breast_MRI_495","Breast_MRI_453","Breast_MRI_479","Breast_MRI_012","Breast_MRI_143","Breast_MRI_800","Breast_MRI_223","Breast_MRI_504","Breast_MRI_693","Breast_MRI_357","Breast_MRI_758","Breast_MRI_916","Breast_MRI_399","Breast_MRI_402","Breast_MRI_060","Breast_MRI_378","Breast_MRI_721","Breast_MRI_418","Breast_MRI_283","Breast_MRI_566","Breast_MRI_879","Breast_MRI_581","Breast_MRI_320","Breast_MRI_080","Breast_MRI_062","Breast_MRI_689","Breast_MRI_796","Breast_MRI_631","Breast_MRI_531","Breast_MRI_597","Breast_MRI_287","Breast_MRI_358","Breast_MRI_815","Breast_MRI_396","Breast_MRI_281","Breast_MRI_910","Breast_MRI_824","Breast_MRI_301","Breast_MRI_165","Breast_MRI_397","Breast_MRI_467","Breast_MRI_868","Breast_MRI_620","Breast_MRI_044","Breast_MRI_400","Breast_MRI_277","Breast_MRI_880","Breast_MRI_353","Breast_MRI_594","Breast_MRI_409","Breast_MRI_090","Breast_MRI_628","Breast_MRI_248","Breast_MRI_912","Breast_MRI_766","Breast_MRI_488","Breast_MRI_660","Breast_MRI_225","Breast_MRI_128","Breast_MRI_917","Breast_MRI_745","Breast_MRI_737","Breast_MRI_809","Breast_MRI_018","Breast_MRI_134","Breast_MRI_131","Breast_MRI_857","Breast_MRI_199","Breast_MRI_709","Breast_MRI_432","Breast_MRI_850","Breast_MRI_037","Breast_MRI_632","Breast_MRI_203","Breast_MRI_921","Breast_MRI_789","Breast_MRI_545","Breast_MRI_412","Breast_MRI_247","Breast_MRI_533","Breast_MRI_726","Breast_MRI_724","Breast_MRI_864","Breast_MRI_702","Breast_MRI_425","Breast_MRI_001","Breast_MRI_486","Breast_MRI_137","Breast_MRI_020","Breast_MRI_922","Breast_MRI_155","Breast_MRI_530","Breast_MRI_712","Breast_MRI_213","Breast_MRI_528","Breast_MRI_158","Breast_MRI_485","Breast_MRI_465"]
print("WARNING: Using VALIDATIONSET_LIST and TESTSET_LIST from MAEMI paper.") #up until 15022024 19:58h

# The below validation and testset lists are for the Duke dataset based on segmentation masks from Caballo et al
#VALIDATIONSET_LIST = ['Breast_MRI_001','Breast_MRI_002','Breast_MRI_005','Breast_MRI_010','Breast_MRI_012','Breast_MRI_019','Breast_MRI_021','Breast_MRI_022','Breast_MRI_028','Breast_MRI_032','Breast_MRI_043','Breast_MRI_044','Breast_MRI_051','Breast_MRI_055','Breast_MRI_057','Breast_MRI_059','Breast_MRI_060','Breast_MRI_061','Breast_MRI_069','Breast_MRI_071','Breast_MRI_077','Breast_MRI_082','Breast_MRI_091','Breast_MRI_099','Breast_MRI_101','Breast_MRI_103','Breast_MRI_104','Breast_MRI_105','Breast_MRI_107','Breast_MRI_114','Breast_MRI_115','Breast_MRI_116','Breast_MRI_117','Breast_MRI_119','Breast_MRI_120','Breast_MRI_123','Breast_MRI_129','Breast_MRI_132','Breast_MRI_134','Breast_MRI_136','Breast_MRI_137','Breast_MRI_141','Breast_MRI_142','Breast_MRI_144','Breast_MRI_150','Breast_MRI_156','Breast_MRI_157','Breast_MRI_160','Breast_MRI_167','Breast_MRI_168','Breast_MRI_176','Breast_MRI_177','Breast_MRI_178','Breast_MRI_180','Breast_MRI_185','Breast_MRI_189','Breast_MRI_192','Breast_MRI_198','Breast_MRI_202','Breast_MRI_205','Breast_MRI_211','Breast_MRI_218','Breast_MRI_225','Breast_MRI_228','Breast_MRI_233','Breast_MRI_234','Breast_MRI_236','Breast_MRI_237','Breast_MRI_239','Breast_MRI_240','Breast_MRI_244','Breast_MRI_253','Breast_MRI_255','Breast_MRI_258','Breast_MRI_265','Breast_MRI_269','Breast_MRI_271','Breast_MRI_275','Breast_MRI_282','Breast_MRI_283','Breast_MRI_290','Breast_MRI_298','Breast_MRI_301','Breast_MRI_303','Breast_MRI_304','Breast_MRI_306','Breast_MRI_313','Breast_MRI_317','Breast_MRI_323','Breast_MRI_328','Breast_MRI_333','Breast_MRI_338','Breast_MRI_345','Breast_MRI_350','Breast_MRI_353','Breast_MRI_360','Breast_MRI_383','Breast_MRI_386','Breast_MRI_395','Breast_MRI_397','Breast_MRI_398','Breast_MRI_399','Breast_MRI_400','Breast_MRI_407','Breast_MRI_408','Breast_MRI_424','Breast_MRI_428','Breast_MRI_429','Breast_MRI_435','Breast_MRI_438','Breast_MRI_441','Breast_MRI_444','Breast_MRI_454','Breast_MRI_457','Breast_MRI_464','Breast_MRI_465','Breast_MRI_468','Breast_MRI_474','Breast_MRI_486','Breast_MRI_489','Breast_MRI_491','Breast_MRI_501','Breast_MRI_506','Breast_MRI_507','Breast_MRI_508','Breast_MRI_512','Breast_MRI_514','Breast_MRI_521','Breast_MRI_525','Breast_MRI_530','Breast_MRI_534','Breast_MRI_539','Breast_MRI_541','Breast_MRI_543','Breast_MRI_546','Breast_MRI_552','Breast_MRI_558','Breast_MRI_559','Breast_MRI_560','Breast_MRI_562','Breast_MRI_567','Breast_MRI_577','Breast_MRI_585','Breast_MRI_590','Breast_MRI_595','Breast_MRI_597','Breast_MRI_605','Breast_MRI_607','Breast_MRI_609','Breast_MRI_610','Breast_MRI_612','Breast_MRI_614','Breast_MRI_615','Breast_MRI_616','Breast_MRI_623','Breast_MRI_636','Breast_MRI_641','Breast_MRI_645','Breast_MRI_650','Breast_MRI_651','Breast_MRI_652','Breast_MRI_656','Breast_MRI_660','Breast_MRI_663','Breast_MRI_666','Breast_MRI_670','Breast_MRI_672','Breast_MRI_677','Breast_MRI_679','Breast_MRI_686','Breast_MRI_687','Breast_MRI_691','Breast_MRI_693','Breast_MRI_694','Breast_MRI_697','Breast_MRI_718','Breast_MRI_724','Breast_MRI_725','Breast_MRI_735','Breast_MRI_746','Breast_MRI_751','Breast_MRI_754','Breast_MRI_757','Breast_MRI_758','Breast_MRI_762','Breast_MRI_765','Breast_MRI_774','Breast_MRI_775','Breast_MRI_780','Breast_MRI_789','Breast_MRI_790','Breast_MRI_792','Breast_MRI_797','Breast_MRI_804','Breast_MRI_805','Breast_MRI_809','Breast_MRI_812','Breast_MRI_816','Breast_MRI_830','Breast_MRI_831','Breast_MRI_832','Breast_MRI_833','Breast_MRI_834','Breast_MRI_836','Breast_MRI_839','Breast_MRI_847','Breast_MRI_850','Breast_MRI_860','Breast_MRI_865','Breast_MRI_869','Breast_MRI_873','Breast_MRI_874','Breast_MRI_879','Breast_MRI_882','Breast_MRI_883','Breast_MRI_884','Breast_MRI_885','Breast_MRI_886','Breast_MRI_891','Breast_MRI_899','Breast_MRI_914','Breast_MRI_915','Breast_MRI_916','Breast_MRI_917']
#TESTSET_LIST = ['Breast_MRI_009','Breast_MRI_041','Breast_MRI_045','Breast_MRI_048','Breast_MRI_064','Breast_MRI_097','Breast_MRI_163','Breast_MRI_183','Breast_MRI_260','Breast_MRI_268','Breast_MRI_287','Breast_MRI_307','Breast_MRI_356','Breast_MRI_368','Breast_MRI_377','Breast_MRI_378','Breast_MRI_387','Breast_MRI_412','Breast_MRI_414','Breast_MRI_431','Breast_MRI_568','Breast_MRI_618','Breast_MRI_633','Breast_MRI_640','Breast_MRI_642','Breast_MRI_662','Breast_MRI_684','Breast_MRI_778','Breast_MRI_799','Breast_MRI_907']
#print("WARNING: Using VALIDATIONSET_LIST and TESTSET_LIST from Pre_Post_Synthesis paper.")

def get_aspect_ratios(nifti_file_header):
    # Calculate the aspect ratios based on pixdim values extracted from the header
    pix_dim = nifti_file_header['pixdim'][1:4]
    aspect_ratios = [pix_dim[1] / pix_dim[2], pix_dim[0] / pix_dim[2], pix_dim[0] / pix_dim[1]]
    if VERBOSE: print('The required aspect ratios are: ', aspect_ratios)
    return aspect_ratios, pix_dim


def get_new_scan_dims(nifti_file_array, pix_dim, is_mulitply=True):
    # Calculate new image dimensions based on the aspect ratio
    if is_mulitply:
        new_dims = np.multiply(nifti_file_array.shape, pix_dim)
    else:
        new_dims = np.divide(nifti_file_array.shape, pix_dim)
    new_dims = (round(new_dims[0]), round(new_dims[1]), round(new_dims[2]))
    if VERBOSE: print('The new scan dimensions are: ', new_dims)
    return new_dims


def display_middle_slice(scan_array, plane=['axial'], aspect_ratios=[1, 1, 1]):
    scan_array_shape = scan_array.shape
    # Display scan array's middle slices
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f'Scan Array (Middle Slices): {plane} \n aspect ratios: {aspect_ratios}')
    if 'sagital' in plane:
        axs[0].imshow(scan_array[scan_array_shape[0] // 2, :, :], aspect=aspect_ratios[0], cmap='gray')
    if 'coronal' in plane:
        axs[1].imshow(scan_array[:, scan_array_shape[1] // 2, :], aspect=aspect_ratios[1], cmap='gray')
    if 'axial' in plane:
        axs[2].imshow(scan_array[:, :, scan_array_shape[2] // 2], aspect=aspect_ratios[2], cmap='gray')
    fig.tight_layout()
    plt.show()


def nifti_to_png(filepath, target_folder, filename, patient_id, is_rotated=True, is_normalised=True):
    try:
    # Load the NIfTI scan and extract data using nibabel
        nifti_file = nib.load(filepath)
        nifti_file_array = nifti_file.get_fdata()
        if VERBOSE:
            print(f"File '{filename}' for patient_id '{patient_id}' was loaded.")
    except Exception as e:
        #print(f"File '{filename}' for patient_id '{patient_id}' could not be loaded. Now continuing with next file.. Error: {e} ")
        return None
    aspect_ratios, pix_dim = get_aspect_ratios(nifti_file.header)
    # Explore the data a bit if verbose
    if VERBOSE:
        print('The nifti header is as follows: \n', nifti_file.header)
        display_middle_slice(nifti_file_array)
        display_middle_slice(nifti_file_array, aspect_ratios=aspect_ratios)

    if USE_BOX_ANNOTATIONS:
        # New SLIDE_MIN and new SLIDE_MAX calculation based on box annotations
        total_number_of_slices_in_mri = nifti_file_array.shape[2]

        # get the total number of colums in the nifti_file_array (i.e. the number of columns in the axial view)
        total_number_of_column_in_mri = nifti_file_array.shape[1]
        total_number_of_rows_in_mri = nifti_file_array.shape[0]

        #print(f"total_number_of_slices_in_mri: {total_number_of_slices_in_mri}, patient_id: {patient_id}")
        assert patient_id in box_annotations_dict, f"Error: Patient with id '{patient_id}' was not found in box_annotations_dict! Please verify why."
        start_slice, end_slice, start_row, end_row, start_column, end_column = box_annotations_dict[patient_id]

        # Extracting the correct slices from the MRI volume as mentioned in IMAGE ANNOTATIONS in https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/
        SLIDE_MIN = total_number_of_slices_in_mri - (end_slice+1)
        SLIDE_MAX = total_number_of_slices_in_mri - (start_slice+1)
        SLIDE_MIN = SLIDE_MIN - round(total_number_of_slices_in_mri * BOX_AXIAL_MARGIN_PERCENTAGE)
        SLIDE_MAX = SLIDE_MAX + round(total_number_of_slices_in_mri * BOX_AXIAL_MARGIN_PERCENTAGE)
        if SLIDE_MIN < 0:
            SLIDE_MIN = 0
        if SLIDE_MAX > total_number_of_slices_in_mri:
            SLIDE_MAX = total_number_of_slices_in_mri
        #print(f"total_slices_in_mri: {total_number_of_slices_in_mri}, patient_id: {patient_id}, "
        #      f"start_slice: {start_slice}, end_slice: {end_slice}, SLIDE_MIN: {SLIDE_MIN_initial}, SLIDE_MAX: {SLIDE_MAX_initial}. "
        #      f"With {BOX_AXIAL_MARGIN_PERCENTAGE*100}% margins, new SLIDE_MIN: {SLIDE_MIN}, and new SLIDE_MAX: {SLIDE_MAX}")
        #start_column_initial = start_column
        #start_column = total_number_of_column_in_mri - (end_column)
        #end_column = total_number_of_column_in_mri - (start_column_initial)
        # A manual check revealed that while the row annotations are correct, the column annotations (a) need to be transformed to a_max-a # r box annotations are not always correct.
        #start_row_initial = start_row
        #start_row = total_number_of_rows_in_mri - (end_row)
        #end_row = total_number_of_rows_in_mri - (start_row_initial)

    # Get the new scan dimensions based on the aspect ratios
    new_dims = get_new_scan_dims(nifti_file_array=nifti_file_array, pix_dim=pix_dim)
    if is_normalised:
        if "09" in filename and VERBOSE:
            # checking axial view of slide 80  to see voxel value ranges e.g. for scan 9.
            print(
                f'{filename} axial view pixel values: '
                f'max={np.max(nifti_file_array[:, :, :])} '
                f'min={np.min(nifti_file_array[:, :, :])} '
                f'mean={np.mean(nifti_file_array[:, :, :])}, '
                f'std={np.std(nifti_file_array[:, :, :])}')

        # Normalise image array to range [0, 255]
        nifti_file_array = ((nifti_file_array - np.min(nifti_file_array)) / (
                np.max(nifti_file_array) - np.min(nifti_file_array))) * 255.0

        if "09" in filename and VERBOSE:
            # checking 3D MRI voxel values to see normalised values e.g. for scan 0009.
            print(
                f'{filename} NORMALISED axial view pixel values: '
                f'max={np.max(nifti_file_array[:, :, :])} '
                f'min={np.min(nifti_file_array[:, :, :])} '
                f'mean={np.mean(nifti_file_array[:, :, :])}, '
                f'std={np.std(nifti_file_array[:, :, :])}')

    # Iterate over nifti_file_array in either coronal, axial or sagital view to extract slices
    for view in VIEWS:
        if view == 'sagital':
            idx = 0
        elif view == 'coronal':
            idx = 1
        elif view == 'axial':
            idx = 2
        for i in range(nifti_file_array.shape[idx]):
            if FOR_PIX2PIX and (i < SLIDE_MIN or i > SLIDE_MAX):
                continue
            try:
                if view == 'sagital':
                    img = nifti_file_array[i, :, :]
                    # img = np.flip(img, axially=0)
                    # As we are norm shrinking the image, INTER_AREA interpolation is preferred.
                    img = extract_tumor(img, start_column, end_column, start_row,
                                        end_row, total_number_of_rows_in_mri, total_number_of_column_in_mri, case=filepath) if EXTRACT_TUMOR_ROI_ONLY else img

                    if not EXTRACT_SINGLE_BREAST_QUADRATIC_ROI:
                        # We avoid resizing (which may cause some errors after tumor ROI cropping)
                        img = cv2.resize(img, (new_dims[2], new_dims[1]), interpolation=cv2.INTER_AREA)
                elif view == 'coronal':
                    img = nifti_file_array[:, i, :]
                    # img = np.flip(img, axis=0)
                    img = extract_tumor(img, start_column, end_column, start_row,
                                        end_row, total_number_of_rows_in_mri, total_number_of_column_in_mri, case=filepath) if EXTRACT_TUMOR_ROI_ONLY else img
                    if not EXTRACT_SINGLE_BREAST_QUADRATIC_ROI:
                        # We avoid resizing (which may cause some errors after tumor ROI cropping)
                        img = cv2.resize(img, (new_dims[2], new_dims[0]), interpolation=cv2.INTER_AREA)
                elif view == 'axial':
                    if '120' in patient_id and VERBOSE:
                        print(f"A: nifti_file_array.shape={nifti_file_array.shape}")

                    img = nifti_file_array[:, :, i]
                    if '120' in patient_id and VERBOSE:
                        print(f"A: img={img}")
                    img = extract_tumor(img, start_column, end_column, start_row,
                                        end_row, total_number_of_rows_in_mri, total_number_of_column_in_mri, case=filepath) if EXTRACT_TUMOR_ROI_ONLY else img
                    if '120' in patient_id and VERBOSE:
                        print(f"B: img={img}")

                    if not EXTRACT_SINGLE_BREAST_QUADRATIC_ROI:
                        # We avoid resizing (which may cause some errors after tumor ROI cropping)
                        img = cv2.resize(img, (new_dims[1], new_dims[0]), interpolation=cv2.INTER_AREA)

                if is_rotated:
                    # Rotate the image 90 degrees counter-clockwise to get the correct orientation
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Create the target folder if it does not exist
                if FOR_PIX2PIX:
                    target_folder_w_view = target_folder
                else:
                    target_folder_w_view = os.path.join(target_folder, view)
                os.makedirs(target_folder_w_view, exist_ok=True)
                # Save the image as a PNG file
                if RESIZE_TO is not None:
                    assert type(RESIZE_TO) == int and RESIZE_TO > 0, "RESIZE_TO must be an integer greater than 0."
                    # Note: Here we use bicubic interpolation as this is also used internally in Pix2PixHD (get_transform function in base_dataset.py)
                    # For reproducibility, we want to have the same input images (i.e. cv2.INTER_CUBIC resized images) for any generative model that we use
                    # In general, bicubic interpolation is a preferred option for upscaling images.
                    img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_CUBIC)
                if IS_SEGMENTATION:
                    filename_stored = f'{filename.replace(".nii.gz", "_0001")}_mask{i}.png'
                else:
                    filename_stored = f'{filename.replace(".nii.gz", "")}_slice{i}.png'
                cv2.imwrite(os.path.join(target_folder_w_view, filename_stored), img)
            except Exception as e:
                print(f"For image {i} of {filename} an exception occurred during png conversion: {e}")
                continue


def convert_and_save(patient):
    # Set variables for conversion
    patient_id = patient[0]

    # We only want to do the conversion for files that are available in the indicated folder
    segmentation_exists = os.path.exists(os.path.join(INPUT_FOLDER_PATH, patient_id + ".nii.gz"))
    image_folder_exists = os.path.exists(os.path.join(INPUT_FOLDER_PATH, patient_id))
    if image_folder_exists or (segmentation_exists and IS_SEGMENTATION):
        #if VERBOSE:
        print(f"Patient folder or file '{os.path.join(INPUT_FOLDER_PATH, patient_id)}' exist.")
        # Iterate over different T1-Weighted Dynamic Contrast-Enhanced MRI phases/sequences for current patient.

        digit = 1 if IS_SEGMENTATION else 0

        for indx in range(2, len(patient)):
            if patient[indx] != '' or IS_SEGMENTATION:
                # Check if the patient should be in our test set or training set
                # Also, check if the T1-weighted DCE-MRI sequence is relevant for the current task
                # For now, only pre-contrast (=0) and post-contrast 1 (=1) are relevant for pix2pix
                if FOR_PIX2PIX and digit not in DIGITS_TO_STORE:
                    continue
                elif FOR_PIX2PIX and digit == 0:
                    if patient_id in TESTSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[2]
                    elif patient_id in VALIDATIONSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[4]
                    else:
                        if SKIP_TRAINING:
                            continue
                        target_folder = OUTPUT_FOLDER_PATH[0]
                elif FOR_PIX2PIX and digit > 0:
                    if patient_id in TESTSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[3]
                    elif patient_id in VALIDATIONSET_LIST:
                        target_folder = OUTPUT_FOLDER_PATH[5]
                    else:
                        if SKIP_TRAINING:
                            continue
                        target_folder = OUTPUT_FOLDER_PATH[1]

                else:
                    target_folder = os.path.join(OUTPUT_FOLDER_PATH, patient_id)

                if IS_SEGMENTATION:
                    filename = f'{patient_id}.nii.gz'
                    # Convert the NIfTI file to a PNG file and store it in the target folder

                    nifti_to_png(filepath=os.path.join(INPUT_FOLDER_PATH, filename),
                                 target_folder=target_folder,
                                 filename=filename,
                                 patient_id=patient_id,
                                 )
                    break # we only have one segmentation per case (the same mask for all DCE sequences)
                else:
                    filename = f'{patient_id}_000{digit}.nii.gz'
                    print(os.path.join(INPUT_FOLDER_PATH, patient_id, filename))

                    # Convert the NIfTI file to a PNG file and store it in the target folder
                    nifti_to_png(filepath=os.path.join(INPUT_FOLDER_PATH, patient_id, filename),
                                 target_folder=target_folder,
                                 filename=filename,
                                 patient_id=patient_id,
                                 )
                # digit + 1 -> next phase/sequence of the MRI scan
                digit = digit + 1
    else:
        #if VERBOSE:
        print(f"Patient folder '{os.path.join(INPUT_FOLDER_PATH, patient_id)}' does not exist. Skipping patient.")

def extract_tumor(img, start_column, end_column, start_row, end_row, total_number_of_rows_in_mri, total_number_of_column_in_mri, case, test=False, adjust_height=True, adjust_width=True):
        total_number_of_column_in_mri = img.shape[1] if total_number_of_rows_in_mri is None else total_number_of_column_in_mri
        total_number_of_rows_in_mri = img.shape[0] if total_number_of_rows_in_mri is None else total_number_of_column_in_mri
        # Extract the ROI around the tumor based on the box annotations
        # Set the pixels inside the ROI area to white (255)
        # img[start_column:end_column, start_row:end_row] = 10
        # img[start_row:end_row, start_column:end_column] = 60
        column_start = total_number_of_column_in_mri - (end_column)
        column_end = total_number_of_column_in_mri - (start_column)
        row_start = total_number_of_rows_in_mri - (end_row)
        row_end = total_number_of_rows_in_mri - (start_row)

        ## if test
        if img.shape[0]==448:
            x_min = start_column
            x_max = end_column
            y_min = row_start
            y_max = row_end
            test_intensity = 100
        elif img.shape[0]==512:
            x_min = column_start
            x_max = column_end
            y_min = start_row
            y_max = end_row
            test_intensity = 250
        elif img.shape[0]==320:
            x_min = start_column
            x_max = end_column
            y_min = row_start
            y_max = row_end
            test_intensity = 0
            if VERBOSE:
                print(f"Warning: For image {case} (shape: {img.shape}).")
        else:
            #raise error
            raise Exception(f"Error: img.shape[0] is neither 448 nor 512 nor 320: {img.shape[0]}. Case: {case}")

        if EXTRACT_SINGLE_BREAST_QUADRATIC_ROI:
            x_min_init = x_min
            x_max_init = x_max
            y_min_init = y_min
            y_max_init = y_max

            if adjust_height:
                # Redefine the height_max and x_min to get an image with 1/2 area of the entire slice. This is done to get a rectangular ROI containing tumor.
                if (y_max - y_min) < (img.shape[0]/2):
                    missing_diff = img.shape[0]/2 - (y_max - y_min)
                    y_max = y_max + missing_diff/2
                    y_min = y_min - missing_diff/2
                    if not float(y_max).is_integer():
                        # round up to the nearest integer
                        y_max = math.ceil(y_max)
                    if not float(y_min).is_integer():
                        # round down to the nearest integer
                        y_min = math.floor(y_min)
                else:
                    # log a warning that tumor annotation larger than half of height of image
                    print(f"Warning: For image (shape: {img.shape}). "
                          f"Tumor annotation larger than half of height of image "
                          f"(y_min={y_min}, y_max={y_max}). Case: {case}")
                if y_max > img.shape[0]:
                    residual_diff = y_max - img.shape[0]
                    y_max = img.shape[0]
                    y_min = y_min - residual_diff
                if y_min < 0:
                    residual_diff = abs(y_min)
                    y_min = 0
                    # Note: Tested assumption: No tumor annotations are that close to both edges of the image that we would need to test again if height_max > img.shape[0]
                    y_max = y_max + residual_diff

            if adjust_width:
                # Redefine the y_max and y_min to get an image with 1/4 area of the entire slice.
                # This is done by cropping only the half (either left or right breast) that contains the tumor resulting in quadratic rectangle of either 224x224 (448 initially) or 256x256 (512 initially).
                if x_max > img.shape[1]/2 and x_min > img.shape[1]/2:
                    # we are interested in the right breast (i.e. the half of the image that contains the tumor)
                    # let's get the entire right half of that mri slice
                    x_max = img.shape[1]
                    x_min = img.shape[1]/2
                elif x_max < img.shape[1]/2 and x_min < img.shape[1]/2:
                    # we are interested in the left breast (i.e. the half of the image that contains the tumor)
                    x_max = img.shape[1]/2
                    x_min = 0
                elif x_max > img.shape[1]/2 and x_min < img.shape[1]/2:
                    # It is not clear if the tumor is in the right or left breast in this image. We crop the breast that contains more annotated area.
                    # we log this as a warning showing which case and what are the shapes
                    if VERBOSE:
                        print(f"Warning: For image (shape: {img.shape}). "
                              f"It is not clear if the tumor is in the right or left breast in this image "
                              f"(x_min={x_min}, x_max={x_max}). "
                              f"We crop the breast that contains more annotated area while ensuring all of tumor bb is in cropped image. Case: {case}")
                    if (x_max - x_min) > img.shape[1]/2:
                        # the right breast contains more annotated area
                        # let's get the entire right half of that mri slice
                        x_max = x_min + img.shape[1]/2
                    else:
                        # the left breast contains more annotated area
                        x_min = x_max - img.shape[1]/2

            x_max = int(x_max)
            x_min = int(x_min)
            y_max = int(y_max)
            y_min = int(y_min)

        # log the extracted tumor ROI
        #if VERBOSE:
        if VERBOSE:
            print(f"Extracted tumor ROI: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}. Case: {case}")
        if test:
            # if img.shape[0] == 320:
            #     x_min = start_column
            #     x_max = end_column
            #     y_min = row_start
            #     y_max = row_end
            #     img = img[x_min:x_max, y_min:y_max] #= 0 # Black
            #     #x_min = column_start
            #     #x_max = column_end
            #     #y_min = start_row
            #     #y_max = end_row
            #     #img = img[x_min:x_max, y_min:y_max] #= 255 # White
            # else:
            img[x_min:x_max, y_min:y_max] = test_intensity
            img[x_min_init:x_max_init, y_min_init:y_max_init] = test_intensity - 100
            img[x_min_init:x_max_init, y_min_init:y_max_init] = test_intensity - 100
        elif EXTRACT_BOUNDING_BOX_MASK:
            img = np.zeros_like(img)
            img[x_min:x_max, y_min:y_max] = 255
        else:
            img = img[x_min:x_max, y_min:y_max]

        # img[start_row:end_row, start_column:end_column] = 160
        # A manual check revealed that while the row annotations are correct, the column annotations (a) need to be transformed to a_max-a # r box annotations are not always correct.

        # img[start_column:end_column, start_row:end_row] = 210
        # img[start_row:end_row, start_column:end_column] = 255
        #
        # Resize to specified size
        if EXTRACT_TUMOR_ROI_SIZE is not None:
            img = cv2.resize(img, (EXTRACT_TUMOR_ROI_SIZE, EXTRACT_TUMOR_ROI_SIZE), interpolation=cv2.INTER_AREA)
        return img

# with open('/home/riti/Radioval_data/Duke_Breast_MRI_all_phases.csv') as file_obj:
for idx, csv_path in enumerate(CSV_PATHS):
    if IS_SEGMENTATION:
        pass
    elif idx == 0 and not len(CSV_PATHS) == 1:
        INPUT_FOLDER_PATH = os.path.join(PREFIX_PATH, 'Duke-Breast-Cancer-MRI-Nifti-Whole')
    else:
        INPUT_FOLDER_PATH = os.path.join(PREFIX_PATH2, 'Duke-Breast-Cancer-MRI-Nifti-Whole')
    #print(f"INPUT_FOLDER_PATH: {INPUT_FOLDER_PATH}")
    with open(csv_path) as file_obj:
        reader_obj = csv.reader(file_obj)
        for row in tqdm(reader_obj):
            convert_and_save(row)
print(f'Done. Extracted all NIfTI files as PNG files in {OUTPUT_FOLDER_PATH}.')

# if adjust_height:
#     # Redefine the height_max and height_min to get an image with 1/2 area of the entire slice. This is done to get a rectangular ROI containing tumor.
#     if (height_max - height_min) < (img.shape[0] / 2):
#         missing_diff = img.shape[0] / 2 - (height_max - height_min)
#         height_max = height_max + missing_diff / 2
#         height_min = height_min - missing_diff / 2
#         if not float(height_max).is_integer():
#             # round up to the nearest integer
#             height_max = math.ceil(height_max)
#         if not float(height_min).is_integer():
#             # round down to the nearest integer
#             height_min = math.floor(height_min)
#     else:
#         # log a warning that tumor annotation larger than half of height of image
#         print(f"Warning: For image (shape: {img.shape}). "
#               f"Tumor annotation larger than half of height of image "
#               f"(height_min={height_min}, height_max={height_max}). Case: {case}")
#     if height_max > img.shape[0]:
#         residual_diff = height_max - img.shape[0]
#         height_max = img.shape[0]
#         height_min = height_min - residual_diff
#     if height_min < 0:
#         residual_diff = abs(height_min)
#         height_min = 0
#         # Note: Tested assumption: No tumor annotations are that close to both edges of the image that we would need to test again if height_max > img.shape[0]
#         height_max = height_max + residual_diff
#
# if adjust_width:
#     # Redefine the width_max and width_min to get an image with 1/4 area of the entire slice.
#     # This is done by cropping only the half (either left or right breast) that contains the tumor resulting in quadratic rectangle of either 224x224 (448 initially) or 256x256 (512 initially).
#     if width_max > img.shape[1] / 2 and width_min > img.shape[1] / 2:
#         # we are interested in the right breast (i.e. the half of the image that contains the tumor)
#         # let's get the entire right half of that mri slice
#         width_max = img.shape[1]
#         width_min = img.shape[1] / 2
#     elif width_max < img.shape[1] / 2 and width_min < img.shape[1] / 2:
#         # we are interested in the left breast (i.e. the half of the image that contains the tumor)
#         width_max = img.shape[1] / 2
#         width_min = 0
#     elif width_max > img.shape[1] / 2 and width_min < img.shape[1] / 2:
#         # It is not clear if the tumor is in the right or left breast in this image. We crop the breast that contains more annotated area.
#         # we log this as a warning showing which case and what are the shapes
#         if VERBOSE:
#             print(f"Warning: For image (shape: {img.shape}). "
#                   f"It is not clear if the tumor is in the right or left breast in this image "
#                   f"(width_min={width_min}, width_max={width_max}). "
#                   f"We crop the breast that contains more annotated area while ensuring all of tumor bb is in cropped image. Case: {case}")
#         if (width_max - width_min) > img.shape[1] / 2:
#             # the right breast contains more annotated area
#             # let's get the entire right half of that mri slice
#             width_max = width_min + img.shape[1] / 2
#         else:
#             # the left breast contains more annotated area
#             width_min = width_max - img.shape[1] / 2