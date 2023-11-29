import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm

from Code import datafile_folder

dfs = []
for src in ['train', 'valid', 'test']:
    # Folder containing Nifti files
    data_folder = os.path.join(datafile_folder, src, 'imgs')

    # Get a list of all Nifti files in the folder
    nifti_files = [file for file in os.listdir(data_folder) if file.endswith(".nii.gz")]

    # Create a list to store file names and image shapes
    data_list = []

    # Initialize a dictionary to store the maximum values for each dimension
    max_dimensions = {"x": 0, "y": 0, "z": 0}

    # Load Nifti files and extract file name, image shape, and update maximum values
    for file in nifti_files:
        file_path = os.path.join(data_folder, file)
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)

        # Update maximum values for each dimension
        max_dimensions["x"] = max(max_dimensions["x"], image_array.shape[0])
        max_dimensions["y"] = max(max_dimensions["y"], image_array.shape[1])

        file_info = {"FileName": file, "ImageShape": image_array.shape}
        data_list.append(file_info)

    # Create a DataFrame with the maximum values for each dimension
    max_dimensions_df = pd.DataFrame([max_dimensions])

    # Convert the list to a Pandas DataFrame
    df = pd.DataFrame(data_list)
    dfs.append(df)
    # Print the DataFrame
    # print(df)

    # Print the DataFrame
    print(max_dimensions_df)
