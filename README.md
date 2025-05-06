
# ⚽ Football Image Processing with OpenCV

This project focuses on **image processing in football (soccer)** using Python and OpenCV. It aims to analyze and process football-related images (e.g., detecting the ball, players, field boundaries, etc.) for computer vision applications such as:
- Player tracking
- Ball detection
- Field segmentation

## 📦 Features

- Load and process football match images
- Convert between color spaces (RGB, HSV, etc.)
- Apply filters and edge detection (e.g., Canny)
- Detect and highlight football objects (ball, players)
- Basic object detection using contour or thresholding techniques

## 📁 Files

```
main.py        # Main script to process football images
output/        # Folder where processed images will be saved
```

## 🚀 Getting Started

### 1. Requirements

Install dependencies with pip:

```bash
pip install opencv-python numpy
```

### 2. Run the Script

```bash
python main.py
```

The script will:
- Load images from a folder
- Process them using selected methods
- Display and/or save the output

## 🧠 Techniques Used

- **Color segmentation (HSV filter)** for detecting the green field or specific jerseys
- **Canny Edge Detection** for contouring
- **Thresholding and Morphology** to isolate objects
- **Contour Analysis** for object detection (ball, players)

## 🖼️ Sample Output

| Original Image | Processed Output |
|----------------|------------------|
| ![](images/sample.jpg) | ![](output/sample_processed.jpg) |

## 🔧 Customize

You can change the processing method in `main.py`:
- Adjust HSV ranges to match different team jerseys
- Modify kernel sizes for blurring/morphology
- Use different input images