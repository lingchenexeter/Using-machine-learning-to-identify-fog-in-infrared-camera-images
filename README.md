# Fog Detection and Image Analysis Script

This Python script processes images to detect fog and extract date/time information using a combination of machine learning and optical character recognition (OCR) techniques.

## Features

- Fog detection using a custom CNN model (FogNet)
- Date/time extraction from images using OCR
- Batch processing of images in a specified folder
- Results output to both console and a text file

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- OpenCV (cv2)
- EasyOCR
- tqdm
- Pillow (PIL)

You can install the required packages using pip:

```bash
pip install torch torchvision opencv-python easyocr tqdm Pillow
```

## Usage

Ensure you have the trained FogNet model file (.pth) ready.

Run the script:
```bash
python fog_detection_script.py
```

When prompted, enter:
- The path to your FogNet model file (.pth)
- The path to the folder containing the images you want to process

The script will process all images in the specified folder and output results to:

- The console (real-time)
- A 'results.txt' file in the same folder as the processed images

## Output Format
For each image, the script outputs:
```bash
CopyImage: [filename]
Original text: [OCR extracted text]
Converted datetime: [Formatted date and time]
Fog prediction: [Fog/No Fog]
```

## Notes

The script is set up to process images with date/time information in a specific location (coordinates 312,514 to 417,534). Adjust these coordinates in the script if your images have a different layout.
The fog detection model (FogNet) should be trained separately. This script assumes you have a trained model file.
The script uses CPU by default. For faster processing, ensure you have CUDA set up if you're using a NVIDIA GPU.

## Troubleshooting

If you encounter "Model not found" or "Invalid path" errors, double-check the paths you're entering.
For any "Unable to load image" errors, ensure your image files are not corrupted and are in a supported format (.png, .jpg, .jpeg, .bmp, .tiff).
