# Neo Lensa - YOLOv7 + PyQt5

**Neo Lensa** is a desktop application that integrates **YOLOv7** with **PyQt5** for object detection in real-time and offline modes. It allows users to process live camera feeds (webcams or RTSP streams) or analyze pre-recorded video files. Designed for ease of use, Neo Lensa provides a smooth UI for AI-powered object detection.

![Neo Lensa Screenshot](UI_images/UI_img1.png)

![Neo Lensa Screenshot](UI_images/UI_img2.png)


## Features
- **Real-time Detection** – Supports webcam and RTSP streams for live object detection.
- **Offline Mode** – Process pre-recorded videos for analysis.
- **User-Friendly UI** – Built with PyQt5 for an intuitive experience.
- **Custom Model Support** – Load and run custom YOLOv7 models.
- **Efficient Processing** – Optimized for speed and accuracy.

## Installation

### Prerequisites
Ensure you have the following installed:
- Anaconda
- Python 3.9.21
- PyTorch
- YOLOv7 dependencies
- PyQt5
- OpenCV

### Setup
1. Setup anaconda environment and activate it.
   
2. Clone the repository:
   ```bash
   https://github.com/MHaziqOnn21/Neo-Lensa.git
   cd Neo-Lensa

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Install PyTorch:
   ```bash
   # CUDA 11.8
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

5. Download YOLOv7 weights and place them or any other custom YOLOv7 models in the 'ptmodels' folder.

6. Run the application:
   ```bash
   python main.py

### Usage
- **Real-time Mode: Enter a webcam index or RTSP URL to start detection.**
- **Offline Mode: Upload a video file and process it for object detection.**
- **Adjust settings to use a custom YOLOv7 model if needed.**


# This project is still under development
