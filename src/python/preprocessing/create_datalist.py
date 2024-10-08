""" Script to create train, validation and test data lists with paths to dce images. """
import argparse
import logging
from pathlib import Path
import ast
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="project/outputs/ids/", help="Where the datalist csv files will be stored.")
    parser.add_argument("--root_path", type=str, default="DIDYOUSETTHEROOTPATH?", help="Where the data will be found as e.g. {root_path}/train/train_A")
    # Root Path could be:
        # data/RadioVal/train_test
        # /home/...
        # data/examples
    args = parser.parse_args()
    return args


def create_datalist(source_path, target_path, annotation_path=None, which_text=1, metadata_path="data/LDM_metadata.csv", type=''):

    data_list = []
    images_paths_source = sorted(list(source_path.glob("*.png")))
    print(f"images_paths_source: {source_path} and len={len(images_paths_source)}")
    images_paths_target = sorted(list(target_path.glob("*.png")))
    if annotation_path is not None:
        if not annotation_path.exists():
            print(f"Annotation path '{annotation_path}' does not exist. Fallback: Creating datalist without annotations.")
            annotation_path = None
        elif not annotation_path.is_dir():
            print(f"Annotation path '{annotation_path}' is not a directory. Fallback: Creating datalist without annotations.")
            annotation_path = None
        elif len(list(annotation_path.glob("*.png"))) == 0:
            print(f"Annotation path '{annotation_path}' does not contain any png files. Fallback: Creating datalist without annotations.")
            annotation_path = None
        else:
            images_paths_annotation = sorted(list(annotation_path.glob("*.png")))

    metadata_df = pd.read_csv(metadata_path, sep=";")
    print(f"Now reading metadata_df (len={len(metadata_df)} from {metadata_path}: {metadata_df.head()}")

    # iterate over rows in metadata_df
    for index, patient_row in tqdm(metadata_df.iterrows()):

        # get patient id from metadata
        patient_id = patient_row[0]

        # get the report text
        pd.set_option("display.max_colwidth", 10000) # to to_string the full report text, see https://github.com/pandas-dev/pandas/issues/9784#issuecomment-88702683
        report_text = metadata_df[metadata_df["patient_id"] == patient_id][f"text{which_text}"].to_string(index=False)
        if index == 0: print(f"Example report text: {report_text}")

        # get acquisition times

        acquisition_times = metadata_df[metadata_df["patient_id"] == patient_id]["acquisition_times"].to_string(index=False)
        #print(f"index {index}: Acquisition times: {acquisition_times}")

        try:
            acquisition_times = ast.literal_eval(acquisition_times) # list conversion of string representation of list
        except Exception as e:
            print(f"index:{index}. Error while trying to parse acquisition times: {acquisition_times}. Exception: {e}")

        if index == 0: print(f"Example acquisition_times: {acquisition_times}")
        # get the source images that contain the patient id
        source_images = [x for x in images_paths_source if str(patient_id) in str(x)]
        annotation_images = [x for x in images_paths_annotation if str(patient_id) in str(x)] if annotation_path is not None else None

        if index == 0: print(f"Example source_images for patient_id={patient_id}: {source_images} and annotation_images: {annotation_images}")

        for idx, source_image in enumerate(source_images):
            # get the target images (from all DCE sequences) corresponding to the source image
            target_image_paths = []
            for i in range(1, 6): # 6 to be safe, max DCE sequence length was 5 in Duke Dataset
                target_image_name = source_image.name.replace("_0000_",f"_000{i}_")
                for x in images_paths_target:
                    if str(target_image_name) in str(x):
                        target_image_paths.append(x)
            #print(f"here, source_image_dce_sequences = {source_image_dce_sequences}")
            # Now we try to get the target image corresponding to the different phases and to the source image
            for idx2, target_image_path in enumerate(target_image_paths):
                #target_image_path = [x for x in images_paths_target if str(source_image_dce_sequence) in str(x)]
                #print("here2")
                #if target_image_path is not None and len(target_image_path) > 0:
                #print("here3")
                #assert len(target_image_path) == 1, f"Target image path '{target_image_path}' contains several images. One was expected. Please revise why."
                #target_image_path = str(target_image_path[0])
                logging.debug(f"Target image path: {target_image_path}")

                try:
                    # assert whether the target image points to a file
                    assert Path(target_image_path).exists(), f"Target image '{target_image_path}' does not exist. Please revise why."
                    assert f"_000{idx2+1}_" in str(target_image_path), f"Target image '{target_image_path}' seems to not contain the correct phase number '000{idx2+1}'. Please revise."
                except Exception as e:
                    print(f"Error while asserting path of target image with acq time at index={idx2+1} from acquisition_times={acquisition_times}. Target image path: {target_image_path}. Image not added to dataset. Exception: {e}")

                try:
                    # source_image_dce_sequences start with first postcontrast image at index position 0
                    # +1 because acquisition_times[0] is the pre-contrast image, so we to start at acquisition_times[1] to get dce phases
                    acquisition_time = acquisition_times[idx2+1]
                    data_list.append({"patient_id": str(patient_id), "acquisition_time": int(acquisition_time),
                                      "source": str(source_image), "target": str(target_image_path),
                                      "report": str(report_text), "report_raw": str(report_text),
                                      "segmentation_mask": str(
                                          annotation_images[idx]) if annotation_path is not None else None
                                      })
                except Exception as e:
                    print(f"Error while trying to get acquisition time at index={idx2+1} from acquisition_times={acquisition_times}. Target image path: {target_image_path}. Image not added to dataset. Exception: {e}")


    print(f"The {type} data list was assembled and contains {len(data_list)} entries.")
    print(f"Here one example: {data_list[0]}")

    return pd.DataFrame(data_list)


def main(args):
    output_path = Path(args.output_path)
    root_path = Path(args.root_path)
    output_path.mkdir(parents=True, exist_ok=True)

    types = ['train', 'validation', 'test']
    for type_ in types:
        if (output_path / f"{type_}.tsv").exists():
            print(f"File '{output_path / f'{type_}.tsv'}' already exists. Skipping creation of {type_}.tsv. If you want to recreate it, please delete the file first.")
        else:
            source_path = Path(f"{root_path}/{type_}/{type_}_A/")  # Path("data/train/train_A/")
            target_path = Path(f"{root_path}/{type_}/{type_}_B/")  # Path("data/train/train_A/")
            annotation_path = Path(f"{root_path}/{type_}/annotations/")
            data_df = create_datalist(source_path=source_path, target_path=target_path, annotation_path=annotation_path, type=type_)
            data_df.to_csv(output_path / f"{type_}.tsv", index=False, sep="\t")

if __name__ == "__main__":
    args = parse_args()
    main(args)
