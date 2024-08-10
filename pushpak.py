import os
import json
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Tokenizer

# Paths to the dataset folders
json_folder = 'dataset/train/json'
images_folder = 'dataset/train/image'

# Initialize tokenizer
tokenizer = LayoutLMv3Tokenizer.from_pretrained('microsoft/layoutlmv3-base', clean_up_tokenization_spaces=True)

def load_annotations(json_path):
    with open(json_path) as f:
        annotations = json.load(f)
    return annotations

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def create_dataset(json_folder, images_folder):
    images = []
    tokenized_texts = []
    bboxes = []
    
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            annotations = load_annotations(json_path)
            
            # Assuming each JSON file corresponds to a single image
            image_id = json_file.replace('.json', '.png')  # or the appropriate image format
            image_path = os.path.join(images_folder, image_id)
            image = preprocess_image(image_path)

            for entry in annotations["valid_line"]:
                words = entry["words"]
                texts = []
                bboxes_list = []
                
                for word in words:
                    text = word["text"]
                    bbox = word["quad"]
                    
                    # Correctly format bounding box
                    bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                    
                    # Append text and bbox to lists
                    texts.append(text)
                    bboxes_list.append(bbox)
                
                # Tokenize the texts and add padding and truncation
                tokenized_text = tokenizer(texts, boxes=bboxes_list, return_tensors="pt", padding=True, truncation=True)
                
                # Convert images to tensors
                image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)  # Convert HWC to CHW
                images.append(image_tensor)
                tokenized_texts.append(tokenized_text)
                bboxes.append(torch.tensor(bboxes_list, dtype=torch.float))
    
    return images, tokenized_texts, bboxes

class CustomDocumentDataset(Dataset):
    def __init__(self, images, tokenized_texts, bboxes):
        self.images = images
        self.tokenized_texts = tokenized_texts
        self.bboxes = bboxes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        tokenized_text = self.tokenized_texts[idx]
        bbox = self.bboxes[idx]
        return {
            'image': image,
            'tokenized_text': tokenized_text,
            'bbox': bbox
        }

# Create dataset
images, tokenized_texts, bboxes = create_dataset(json_folder, images_folder)
train_dataset = CustomDocumentDataset(images, tokenized_texts, bboxes)
from transformers import LayoutLMv3ForTokenClassification, Trainer, TrainingArguments

# Load the pre-trained model
model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train the model
trainer.train()
    