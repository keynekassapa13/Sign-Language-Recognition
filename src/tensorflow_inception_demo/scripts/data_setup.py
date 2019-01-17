'''
The purpose of this script is to setup data that contains a .csv file with some generic ID for a group of image files,
and their associated labels. This will setup the images in the folder structure required for Inception V3 to process.

Final dataset folders:
    dataset/label/*.jpg
    dataset/label2/*.jpg

    (Where label is a unique label describing a gesture, e.g., 'Sliding Two Fingers Left'

CSV File has the following columns with no headers:
    id  label
    id  label
    ...

Usage:
    Use python3 scripts/data_setup.py -h to see 'help'.
    python3 scripts/data_setup.py --data=path_to_data --labels=path_to_csv
'''

import argparse
import glob
import pandas as pd
import os
import shutil
import sys

sys.path.append('..')
sys.path.append(os.getcwd())

import uuid
from settings.logger_settings import setup_custom_logger


DATA_SETUP_LOG = setup_custom_logger("DATA SETUP")


def setup_data(dataset_name: str, output_root: str, data_path: str, labels_path: str):
    '''
    Setup folder structure that Inception V3 will use to automatically train and subsequently classify images.

    Note that Inception will automatically choose a training/test/validation ratio of 80/10/10.

    Args:
        data_path: Path to image groups with unique IDs as folder names.
        labels_path: Path to a csv document with aforementioned IDs associated with discreet labels.
    '''
    # Create destination for dataset
    dataset_path = os.path.join(output_root, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    # Sub-directories containing images from each recording, folder names are IDs.
    data_folders = [folder.path for folder in os.scandir(data_path) if folder.is_dir()]
    DATA_SETUP_LOG.info(f'Found {len(data_folders)} data folders for dataset.')
    # For each folder, check it against the master labels csv file
    # read CSV
    master_df = pd.read_csv(labels_path,
                            sep=';',
                            names=["ID", "LABEL"])
    for folder in data_folders:
        id = int(os.path.basename(folder))
        DATA_SETUP_LOG.info(f'Working with ID: {id}')
        # Get label associated with id and make dataset sub-folder.
        try:
            try:
                label = master_df.loc[master_df['ID'] == id, 'LABEL'].values[0]
            except IndexError:
                DATA_SETUP_LOG.info(f'Index Miss on ID: {id}')
                continue
            DATA_SETUP_LOG.info(f'Label identified: {label}')
        except TypeError:
            DATA_SETUP_LOG.error(f'Miss on ID: {id}')
            continue
        destination = os.path.join(dataset_path, label)
        DATA_SETUP_LOG.info(f'Destination for images: {destination}')
        if not os.path.exists(destination):
            os.makedirs(destination, exist_ok=True)
        # Move all jpg images in folder to label folder
        for jpg_img in glob.iglob(os.path.join(folder, "*.jpg")):
            img_name = os.path.basename(jpg_img).split('.')[0]
            img_name = img_name + str(uuid.uuid4()) + ".jpg"
            file_name = os.path.join(destination, img_name)
            DATA_SETUP_LOG.info(f"Copied image: {file_name}")
            shutil.copy(jpg_img, file_name)
    DATA_SETUP_LOG.info("Setup complete.")
    # Visual check for number of images, output to console.
    dataset_folders = [folder.path for folder in os.scandir(dataset_path) if folder.is_dir()]
    for folder in dataset_folders:
        DATA_SETUP_LOG.info(f'Label: "{os.path.basename(folder)}" has {len(os.listdir(folder))} images.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Setup for Inception V3")
    parser.add_argument('--data', help='Path to dataset containing images.', required=True)
    parser.add_argument('--labels', help='CSV file containing image ID, and label.', required=True)
    parser.add_argument('--output', help='Output root folder.', required=True)
    parser.add_argument('--name', help='Name of this resultant dataset.', required=True)
    args = vars(parser.parse_args())
    DATA_SETUP_LOG.info(f'Dataset Folder: {args["data"]}')
    DATA_SETUP_LOG.info(f'Labels CSV: {args["labels"]}')
    DATA_SETUP_LOG.info(f'Output: {args["output"]}')
    DATA_SETUP_LOG.info(f'Dataset Name: {args["name"]}')
    setup_data(args['name'], args['output'], args['data'], args['labels'])
