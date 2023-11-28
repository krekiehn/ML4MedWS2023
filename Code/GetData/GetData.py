import requests
import zipfile
import os

from Code import data_folder


def download_and_unzip(zip_url, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Extract the file name from the URL
    file_name = os.path.join(destination_folder, zip_url.split("/")[-1])

    # Download the zip file
    response = requests.get(zip_url)
    with open(file_name, 'wb') as zip_file:
        zip_file.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # Remove the downloaded zip file
    os.remove(file_name)


if __name__ == "__main__":
    # Replace 'your_zip_file_url' with the actual URL of the zip file
    zip_file_url = 'https://cloud.rz.uni-kiel.de/index.php/s/WRM33znRQjLcMZo/download'

    # Replace 'your_destination_folder' with the desired destination folder
    # print(current_directory := os.getcwd())
    # print(p1 := os.path.dirname(current_directory))
    # print(p2 := os.path.dirname(p1))
    #
    # destination_folder = os.path.join(p2, 'data', 'ConvertedPelvisNiftiDataset')

    download_and_unzip(zip_file_url, data_folder)
