import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import easyocr
from tqdm import tqdm
import re

# Define the FogNet model
class FogNet(nn.Module):
    def __init__(self):
        super(FogNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to predict fog in an image with probabilities
def predict_fog(model, image_path, transform):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).squeeze(0)  # Get probabilities for each class
        _, predicted = torch.max(output, 1)  # Get the class with the highest probability
    
    fog_prob = probabilities[1].item()  # Probability for "Fog"
    no_fog_prob = probabilities[0].item()  # Probability for "No Fog"
    
    prediction = "Fog" if predicted.item() == 1 else "No Fog"
    return prediction, fog_prob, no_fog_prob

# Function to convert datetime
def convert_datetime(text):
    try:
        date_part, time_part = text.split()
        month, day, year = date_part.split('/')
        month = month
        day = day.zfill(2)
        hour, minute, second = time_part.split('.')
        hour = hour.zfill(2)
        minute = minute.zfill(2)
        second = second.zfill(2)
        return f"{month}/{day}/{year} {hour}:{minute}:{second}"
    except Exception:
        try:
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', text)
            if not date_match:
                raise ValueError("Unable to recognize date format")
            date_part = date_match.group(1)
            month, day, year = date_part.split('/')
            month = month
            day = day.zfill(2)
            time_digits = re.findall(r'\d{2}', text[date_match.end():])
            if len(time_digits) < 3:
                raise ValueError("Not enough time digits recognized")
            hour = time_digits[0]
            minute = time_digits[1]
            second = time_digits[2]
            return f"{month}/{day}/{year} {hour}:{minute}:{second}"
        except Exception as e:
            return f"Error: Unable to parse datetime - {str(e)}"

# Function to get valid model path
def get_valid_model_path():
    while True:
        model_path = input("Please enter the model file path: ")
        if os.path.isfile(model_path):
            return model_path
        else:
            print("Model not found, please try again.")

# Function to get valid folder path
def get_valid_folder_path():
    while True:
        folder_path = input("Please enter the image folder path: ")
        if os.path.isdir(folder_path):
            return folder_path
        else:
            print("Invalid path, please try again.")

# Function to perform OCR with error checking and parameter adjustment
def perform_ocr(reader, image, filename):
    initial_params = {
        'detail': 0,
        'paragraph': True,
        'allowlist': '0123456789/: ',
        'contrast_ths': 0.2,
        'adjust_contrast': 0.5
    }
    
    result = reader.readtext(image, **initial_params)
    recognized_text = ' '.join(result)
    converted_datetime = convert_datetime(recognized_text)
    
    if "Error" in converted_datetime:
        print(f"Error detected in {filename}, trying alternative parameters...")
        alternative_params = {
            'detail': 0,
            'paragraph': False,
            'allowlist': '0123456789/: ',
            'contrast_ths': 0.1,
            'adjust_contrast': 0.7
        }
        result = reader.readtext(image, **alternative_params)
        recognized_text = ' '.join(result)
        converted_datetime = convert_datetime(recognized_text)
    
    return recognized_text, converted_datetime

# Natural sort key function
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# Main function
def main():
    # Load the model
    model = FogNet()
    model_weights_path = get_valid_model_path()
    model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    model.eval()

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Get valid folder path
    folder_path = get_valid_folder_path()

    # Get all image files and sort them using natural sort
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort(key=natural_sort_key)

    # Create output file
    output_file = os.path.join(folder_path, 'results.txt')

    # Process images
    results = []
    for filename in tqdm(image_files, desc="Processing"):
        image_path = os.path.join(folder_path, filename)
        
        # Read image for OCR
        image = cv2.imread(image_path)
        
        # Check image size and set crop coordinates
        height, width = image.shape[:2]
        if height == 534 and width == 768:
            x1, y1, x2, y2 = 312, 514, 417, 534
        elif height == 502 and width == 728:
            x1, y1, x2, y2 = 292, 485, 440, 500
        else:
            print(f"Warning: Unexpected image size for {filename}: {width}x{height}")
            continue  # Skip images with unexpected sizes
        
        # Crop image
        cropped_image = image[y1:y2, x1:x2]
        
        # Perform OCR with error checking and parameter adjustment
        recognized_text, converted_datetime = perform_ocr(reader, cropped_image, filename)
        
        # Predict fog
        fog_prediction, fog_prob, no_fog_prob = predict_fog(model, image_path, transform)
        
        # Prepare results
        result = {
            'filename': filename,
            'text': f"image: {filename}\n"
                    f"Original text: {recognized_text}\n"
                    f"Converted datetime: {converted_datetime}\n"
                    f"Fog prediction: {fog_prediction}\n"
                    f"Probability (Fog): {fog_prob:.4f}\n"
                    f"Probability (No Fog): {no_fog_prob:.4f}\n\n"
        }
        
        results.append(result)

    # Sort results by filename using natural sort
    results.sort(key=lambda x: natural_sort_key(x['filename']))

    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result['text'])
            print(result['text'])  # Print results to console

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
