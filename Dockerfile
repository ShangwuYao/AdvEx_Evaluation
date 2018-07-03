# Use an official Python runtime as a parent image
FROM python:2.7-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD ./src /app/src
ADD ./config /app/config
ADD ./__init__.py /app
ADD ./requirements.txt /app
ADD ./Dockerfile /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN  apt-get update \
  && apt-get install -y wget unzip

RUN wget https://s3.amazonaws.com/advex/evaluation_data/image_data_final.zip \
&& mkdir image_data_final/ && unzip image_data_final.zip -d image_data_final \
&& rm image_data_final.zip 

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "./src/evaluation_worker.py"]
