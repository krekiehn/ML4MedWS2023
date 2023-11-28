import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

from Code import datafile_folder


def load_nifti_files(folder_path):
    nifti_files = [file for file in os.listdir(folder_path) if file.endswith('.nii.gz')]

    data_list = []
    for file in nifti_files:
        file_path = os.path.join(folder_path, file)
        img = sitk.ReadImage(file_path)
        data = sitk.GetArrayFromImage(img)
        data_list.append(data)

    return nifti_files, data_list


def get_unique_values_info(data_list):
    unique_values_info = []

    for filename, data in zip(nifti_files, data_list):
        unique_values = np.unique(data)
        num_unique_values = len(unique_values)
        unique_values_info.append({
            'Filename': filename,
            'Unique_Values': unique_values,
            'Num_Unique_Values': num_unique_values
        })

    return unique_values_info


def create_dataframe(unique_values_info):
    df = pd.DataFrame(unique_values_info)
    return df


if __name__ == "__main__":
    for folder in os.listdir(datafile_folder):
        print(folder)
        folder_path = os.path.join(datafile_folder, folder, 'targets')

        # Load NIfTI files
        nifti_files, nifti_data_list = load_nifti_files(folder_path)

        # Get information about unique values
        unique_values_info = get_unique_values_info(nifti_data_list)

        # Create DataFrame
        df = create_dataframe(unique_values_info)

        # Print DataFrame
        print("DataFrame:")
        print(df)

