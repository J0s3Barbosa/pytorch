import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load the labels for COCO dataset
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

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = F.to_tensor
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

# Path to your image
image_path = rf"images"
image = load_image(image_path)

# Perform object detection
with torch.no_grad():
    predictions = model(image)

# Print predictions
print(predictions)

def visualize_predictions(image, predictions, threshold=0.5):
    # Convert the image tensor to PIL Image
    image = F.to_pil_image(image.squeeze(0))

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Loop through the detected objects
    for element in range(len(predictions[0]['boxes'])):
        score = predictions[0]['scores'][element].item()
        if score > threshold:
            box = predictions[0]['boxes'][element].cpu().numpy()
            label = COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'][element].item()]
            # Draw the bounding box
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Add the label
            ax.text(box[0], box[1] - 10, f'{label}: {score:.2f}', color='red', fontsize=12, weight='bold')
    
    plt.axis('off')
    plt.show()

# Visualize the results
visualize_predictions(image, predictions)

