# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /code

# Clone the github repository
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/krekiehn/ML4MedWS2023.git
WORKDIR /code/ML4MedWS2023

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir "/code/ML4MedWS2023/Data"
RUN mkdir "/code/ML4MedWS2023/Data/ConvertedPelvisNiftiDataset"
WORKDIR /code/ML4MedWS2023/Data/ConvertedPelvisNiftiDataset

# Download a file from a URL and unzip it
RUN apt-get install -y curl unzip && \
    curl -LJO https://cloud.rz.uni-kiel.de/index.php/s/WRM33znRQjLcMZo/download && \
    unzip "/code/ML4MedWS2023/Data/ConvertedPelvisNiftiDataset/Converted Pelvis Nifti Dataset.zip" && \
    rm "/code/ML4MedWS2023/Data/ConvertedPelvisNiftiDataset/Converted Pelvis Nifti Dataset.zip" \

# Run script.py when the container launches
#CMD ["python", "./Code/GetData/GetData.py"]
CMD ["python"]