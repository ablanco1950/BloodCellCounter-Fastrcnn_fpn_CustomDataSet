
# asking IA from Bing "python inference for a fastrcnn_fpn mode"
# and adapted and slightly modified by Alfonso Blanco

# Dataset https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu/dataset/3
# model to test https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu

import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO


import matplotlib.pyplot as plt
import os
import re

# PARAMETERS
CONF_THRESHOLD=0.8
SHOW_OPTION = "YES"
imgpath="Blood-Cell-Detection-3\\test\\" # images test folder coco json
# Path to your saved weights file (.pth or .pt)
weights_path = "fasterrcnn_fpn.pth"


LabelClass="blood cell "

# asking IA Bing PIL draw rectangle width changes with rectangle width python
def draw_scaled_rectangle(draw, xy, outline, scale_factor=0.05):
    x0, y0, x1, y1 = xy
    rect_width = abs(x1 - x0)
    rect_height = abs(y1 - y0)
    stroke_width = max(1, int(min(rect_width, rect_height) * scale_factor))
    draw.rectangle(xy, outline=outline, width=stroke_width)

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import json

# -----------------------------
# 1. Custom Dataset Class
# -----------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms

        # Load COCO-style annotations
        with open(annotations_file) as f:
            coco_data = json.load(f)

        self.images_info = coco_data["images"]
        self.annotations = coco_data["annotations"]

        # Map image_id -> list of annotations
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Category mapping
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        filename= img_info["file_name"]
        img = Image.open(img_path).convert("RGB")

        anns = self.image_to_anns.get(img_info["id"], [])

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, w, h = ann["bbox"]
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_info["id"]])
        area = torch.as_tensor([ann["area"] for ann in anns], dtype=torch.float32)
        iscrowd = torch.zeros((len(anns),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target, img_path, filename

    def __len__(self):
        return len(self.images_info)


# -----------------------------
# 2. Data Transforms
# -----------------------------
def get_transform(train):
    transforms = []
    transforms.append(F.to_tensor)
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

# -----------------------------
# 1. Load Pretrained Model
# -----------------------------

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn



# 1️⃣ Create the model architecture (no pretrained weights)
model = fasterrcnn_resnet50_fpn(weights=None)  # For torchvision >= 0.13
# For older versions:
#model = fasterrcnn_resnet50_fpn(pretrained=False)

in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace head for custom classes
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=4)

# 2️⃣ Load the saved state dictionary
try:
    state_dict = torch.load(weights_path, map_location="cpu")  # or "cuda" if GPU
    model.load_state_dict(state_dict)
    print("✅ Weights loaded successfully from", weights_path)
except FileNotFoundError:
    print(f"❌ File not found: {weights_path}")
except RuntimeError as e:
    print(f"❌ Error loading weights: {e}")

model.eval()  # Set to evaluation mode




# -----------------------------
# 2. Load and Preprocess Images
# -----------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

test_images = "Blood-Cell-Detection-3/test"
test_annotations = "Blood-Cell-Detection-3/test/_annotations.coco.json"

dataset_test = CustomDataset(test_images, test_annotations, get_transform(train=False))

data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

ContPlateletsLabeled=0
ContRBCLabeled=0
ContWBCLabeled=0
ContPlateletsPredicted=0
ContRBCPredicted=0
ContWBCPredicted=0

for images, targets, images_path, filenames in data_loader_test:
    
    
    boxes=targets[0]['boxes']
    
    labels=targets[0]['labels']
   
    filename=filenames[0]
    filename=filename[0:16]

    img = Image.open(images_path[0]).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
            #font = ImageFont.truetype("arial.ttf", 16) # MOD
            font_size = 50
            font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
            print("ERROR FONT")
            font = ImageFont.load_default()
    
    
    ContPlatelets=0
    ContRBC=0
    ContWBC=0
    for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            
            TextLabel= str(label.item())
            if TextLabel=="1":
                     draw_scaled_rectangle(draw, (x1, y1, x2, y2), "red",scale_factor=0.05) # Platelets
                     ContPlatelets=ContPlatelets+1
                     
            else:
                if TextLabel=="2":
                     draw_scaled_rectangle(draw, (x1, y1, x2, y2), "blue",scale_factor=0.05) # RBC
                     ContRBC=ContRBC+1
                                         
                else:
                    draw_scaled_rectangle(draw, (x1, y1, x2, y2), "green",scale_factor=0.05) # WBC
                    ContWBC=ContWBC+1
                    
            
            draw.rectangle(((x1, y1), (x1 , y1)), fill="red", width=6)

    ContPlateletsLabeled = ContPlateletsLabeled + ContPlatelets
    ContRBCLabeled=ContRBCLabeled + ContRBC
    ContWBCLabeled = ContWBCLabeled + ContWBC

            
        
    filenameComplete = filename + " Platelets=" +str(ContPlatelets) +  " RBC=" +str(ContRBC) +  " WBC=" +str(ContWBC)
    
       
    # PREDICTIONS

    imgPred = Image.open(images_path[0]).convert("RGB")

    # Transform to tensor
    img_tensor = F.to_tensor(imgPred)  # Converts to [C,H,W] and scales to [0,1]

    # -----------------------------
    # 3. Run Inference
    # -----------------------------
    with torch.no_grad():
         predictions = model([img_tensor])  # Model expects a list of tensors

    # Extract predictions for the first image
    pred = predictions[0]
    boxesPred = pred["boxes"]
    labelsPred = pred["labels"]
    scoresPred = pred["scores"]

    # -----------------------------
    # 4. Filter by Confidence
    # -----------------------------
    threshold = CONF_THRESHOLD
    #print(scoresPred)
    keep = scoresPred >= threshold
    boxesPred = boxesPred[keep]
    labelsPred = labelsPred[keep]
    scoresPred = scoresPred[keep]

    # -----------------------------
    # 5. Draw Results
    # -----------------------------
    drawPred = ImageDraw.Draw(imgPred)
    try:
            #font = ImageFont.truetype("arial.ttf", 16) # MOD
            font_size = 50
            font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
            print("ERROR FONT")
            font = ImageFont.load_default()

    ContPlateletsPred=0
    ContRBCPred=0
    ContWBCPred=0
    for boxPred, labelPred, scorePred in zip(boxesPred, labelsPred, scoresPred):
            x1, y1, x2, y2 = boxPred
            #draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=6)
            #print(labelPred)

            TextLabelPred= str(labelPred.item())
            if TextLabelPred=="1":
                     draw_scaled_rectangle(drawPred, (x1, y1, x2, y2), "red",scale_factor=0.05) # Platelets
                     ContPlateletsPred=ContPlateletsPred+1
            else:
                if TextLabelPred=="2": 
                     draw_scaled_rectangle(drawPred, (x1, y1, x2, y2), "blue",scale_factor=0.05) # RBC
                     ContRBCPred=ContRBCPred+1
                else:
                    draw_scaled_rectangle(drawPred, (x1, y1, x2, y2), "green",scale_factor=0.05) # WB
                    ContWBCPred=ContWBCPred+1 
                       
            drawPred.rectangle(((x1, y1 ), (x1 , y1)), fill="red", width=6)
            
    ContPlateletsPredicted= ContPlateletsPredicted + ContPlateletsPred
    ContRBCPredicted= ContRBCPredicted + ContRBCPred       
    ContWBCPredicted= ContWBCPredicted + ContWBCPred       
   
    filenameCompletePred =  " Platelets=" +str(ContPlateletsPred) +  " RBC=" +str(ContRBCPred) +  " WBC=" +str(ContWBCPred)
      
    if SHOW_OPTION == "YES": 
        fig, axs = plt.subplots(1,2, figsize=(15,5))
        axs[0].imshow(img);      axs[0].set_title('Labeled: ' + filenameComplete);    axs[0].axis('off')
        axs[1].imshow(imgPred); axs[1].set_title('  Predicted: ' + filenameCompletePred ); axs[1].axis('off')
        #plt.tight_layout();
        plt.show()
   
TotBloodCellLabeled=0
TotBloodCellPredicted=0

print(" Platelets labeled = " + str(ContPlateletsLabeled) + " predicted = " + str(ContPlateletsPredicted))
print(" RBC labeled = " + str(ContRBCLabeled) + " predicted = " + str(ContRBCPredicted))
print(" WBC labeled = " + str(ContWBCLabeled) + " predicted = " + str(ContWBCPredicted))      

TotBloodCellLabeled= ContPlateletsLabeled + ContRBCLabeled + ContWBCLabeled      
TotBloodCellPredicted= ContPlateletsPredicted + ContRBCPredicted + ContWBCPredicted

print("")
print("")

print(" Total Blood Cells  labeled = " + str(TotBloodCellLabeled) + " predicted = " + str(TotBloodCellPredicted))      

      

      
