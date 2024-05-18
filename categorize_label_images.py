import json
import os
import shutil
import torch
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
class ImageCategorizer:
    def __init__(self, main_folder, threshold=0.5):
        self.main_folder = main_folder
        self.threshold = threshold
        
        # Load the pre-trained object detection model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def categorize_images(self):
        # Dictionary to keep count of images per category
        image_counts = {}

        # Loop through each file in the main folder
        for filename in os.listdir(self.main_folder):
            filepath = os.path.join(self.main_folder, filename)
            
            if not os.path.isfile(filepath):
                continue
            
            try:
                image = Image.open(filepath).convert("RGB")
            except:
                continue
            
            # Preprocess the image
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0)
            
            # Make a prediction
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Loop through the detected objects
            for element in range(len(predictions[0]['boxes'])):
                score = predictions[0]['scores'][element].item()
                if score > self.threshold:
                    label = COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'][element].item()]
                    
                    # Create the label folder if it doesn't exist
                    label_folder = os.path.join(self.main_folder, label)
                    os.makedirs(label_folder, exist_ok=True)
                    
                    # Update image count for the label
                    if label not in image_counts:
                        image_counts[label] = 1
                    else:
                        image_counts[label] += 1
                    
                    # Create the new filename and move the image
                    new_filename = f'{label}_{image_counts[label]}.jpg'
                    new_filepath = os.path.join(label_folder, new_filename)
                    shutil.move(filepath, new_filepath)
                    print(f'Moved {filename} to {new_filepath}')
                    break  # Move the image after the first high-confidence detection

# Usage example:
main_folder_path = 'images'
categorizer = ImageCategorizer(main_folder_path)
categorizer.categorize_images()
