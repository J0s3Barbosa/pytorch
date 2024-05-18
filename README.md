# Image Categorizer

This project provides a Python class `ImageCategorizer` that uses a pre-trained object detection model to categorize images into folders based on the objects detected in the images. The script dynamically detects objects such as "cat", "dog", "person", etc., creates folders for each detected object, and moves the images into the corresponding folders with renamed filenames.

## Features

- Uses a pre-trained Faster R-CNN model for object detection.
- Dynamically creates folders based on detected object labels.
- Renames and moves images into corresponding folders based on detected objects.
- Handles multiple images in a specified main folder.

## Requirements

- Python 3.6 or later
- `torch` (PyTorch)
- `torchvision`
- `Pillow`

## PyTorch Documentation
https://pytorch.org/
https://pytorch.org/docs/stable/index.html

PyTorch is a fully featured framework for building deep learning models
which is a type of machine learning that's commonly used in applications like image recognition and language processing
uses dynamic computation graphs and is completely Pythonic
allows scientists, developers, and neural network debuggers to run and test portions of the code in real-time

## Installation and Usage
https://pytorch.org/get-started/locally/

1. Clone the repository or download the code files.
2. Install the required Python packages:

py -m venv env
.\env\Scripts\activate
py -m pip install --upgrade pip
pip3 install torch torchvision torchaudio matplotlib requests

add images to the main folder 
run the code
images will be moved toa folder and categorized


