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

# Function to predict fog in an image
def predict_fog(model, image_path, transform):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return "Fog" if predicted.item() == 1 else "No Fog"

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

# Main function
def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get valid model path
    model_weights_path = get_valid_model_path()

    # Load the model
    model = FogNet().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Get valid folder path
    folder_path = get_valid_folder_path()

    # Get all image files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Create output file
    output_file = os.path.join(folder_path, 'results.txt')

    # Process images
    with open(output_file, 'w', encoding='utf-8') as f:
        for filename in tqdm(image_files, desc="Processing"):
            image_path = os.path.join(folder_path, filename)
            
            # Read image for OCR
            image = cv2.imread(image_path)
            
            # Crop image for OCR
            x1, y1, x2, y2 = 291, 485, 437, 534
            cropped_image = image[y1:y2, x1:x2]
            
            # Perform OCR
            result = reader.readtext(cropped_image, detail=0, allowlist='0123456789/. ')
            recognized_text = ' '.join(result)
            
            # Convert datetime
            converted_datetime = convert_datetime(recognized_text)
            
            # Predict fog
            fog_prediction = predict_fog(model, image_path, transform)
            
            # Prepare results
            results = f"image: {filename}\n"
            results += f"Original text: {recognized_text}\n"
            results += f"Converted datetime: {converted_datetime}\n"
            results += f"Fog prediction: {fog_prediction}\n\n"
            
            # Print results to console
            print(results)
            
            # Write results to file
            f.write(results)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()