import os


def FindRepoRoot():
    current_directory = os.getcwd()
    while "ML4MedWS2023" != os.path.basename(current_directory):
        current_directory = os.path.dirname(current_directory)
    return current_directory


print(root_dir := FindRepoRoot())

data_folder = os.path.join(root_dir, 'Data', 'ConvertedPelvisNiftiDataset')
datafile_folder = os.path.join(root_dir, 'Data', 'ConvertedPelvisNiftiDataset', 'Converted Pelvis Nifti Dataset')
