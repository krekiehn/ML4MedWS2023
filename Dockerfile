# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /code

# Clone the github repository
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/krekiehn/ML4MedWS2023.git

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run script.py when the container launches
CMD ["python", "./Code/GetData/GetData.py"]