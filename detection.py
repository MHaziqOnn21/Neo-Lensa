import time
import cv2
import torch
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
import os
from datetime import datetime

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import threading
from threading import Lock


def run_inference(input_queue, output_queue, camera_id_map):
    """
    YOLOv7 inference function.
    Runs in a separate process.
    """
    # Initialize settings (no defaults)
    camera_ids = None
    weights = None
    imgsz = 640  # Default image size (can be adjusted if needed)
    conf_thres = None  # No default confidence threshold
    iou_thres = None  # No default IOU threshold
    device = select_device('')  # Automatically select device (CPU or GPU)
    half = device.type != 'cpu'  # Use half precision if GPU is available

    # Print the device being used
    if device.type == 'cpu':
        print("Using CPU for inference.")
    else:
        # Get the specific GPU name
        gpu_name = torch.cuda.get_device_name(device.index)
        print(f"Using GPU ({gpu_name}) for inference.")

    # Lock for thread-safe parameter updates
    parameter_lock = threading.Lock()

    # Initialize model and datasets (will be set after receiving settings from the frontend)
    model = None
    datasets = None

    # Dictionary to store VideoWriter objects for each camera
    video_writers = {}

    # Flag to indicate if thresholds have been received
    thresholds_received = False

    # Flag to indicate if recording is enabled
    is_recording_enabled = True  # Default: Recording is enabled

    try:
        while True:
            # Debug: Confirm that the backend process is running
            print("Backend process is running...")

            # Check for new settings or stop signal from the frontend
            if not input_queue.empty():
                new_settings = input_queue.get()
                if new_settings == "stop":
                    print("Stop signal received. Exiting...")
                    break  # Exit the loop
                elif isinstance(new_settings, tuple):
                    if new_settings[0] == "update_thresholds":
                        # Update confidence and IOU thresholds
                        with parameter_lock:
                            _, conf_thres, iou_thres = new_settings
                            thresholds_received = True  # Set flag to True
                            print(f"Thresholds received from frontend: Conf={conf_thres}, IOU={iou_thres}")
                    elif new_settings[0] == "update_model":
                        # Update model, camera IDs, and recording state
                        with parameter_lock:
                            _, weights, camera_ids, is_recording_enabled = new_settings
                            print(f"Recording state: {'Enabled' if is_recording_enabled else 'Disabled'}")

                            try:
                                # Load the new model
                                model = attempt_load(weights, map_location=device)
                                stride = int(model.stride.max())
                                imgsz = check_img_size(imgsz, s=stride)
                                if half:
                                    model.half()
                                names = model.module.names if hasattr(model, 'module') else model.names
                                colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

                                # Reinitialize datasets for new camera IDs
                                datasets = []
                                for camera_id in camera_ids:
                                    try:
                                        dataset = LoadStreams(camera_id, img_size=imgsz, stride=stride)
                                        datasets.append(dataset)
                                        print(f"Camera {camera_id} initialized successfully.")

                                        # Get the first frame to determine frame dimensions and FPS
                                        for path, img, im0s, vid_cap in dataset:
                                            if im0s is not None:
                                                if isinstance(im0s, np.ndarray):  # Single camera
                                                    frame_height, frame_width = im0s.shape[:2]
                                                else:  # Multiple cameras or webcam
                                                    frame_height, frame_width = im0s[0].shape[:2]
                                                break  # Exit after getting the first frame

                                        # Map camera ID to Cam1 or Cam2
                                        cam_id_display = camera_id_map.get(camera_id, f"Cam{camera_id}")

                                        # Create directory structure
                                        now = datetime.now()
                                        month_year = now.strftime("%b_%y")  # e.g., Feb_25
                                        date = now.strftime("%d-%b-%Y")  # e.g., 25-Feb-2025
                                        time_str = now.strftime("%H.%M.%S")  # e.g., 16.10.20
                                        save_dir = os.path.join("RealtimeArchive", month_year, date)
                                        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

                                        # Generate output file path
                                        output_file = os.path.join(save_dir, f"{cam_id_display}_{time_str}.avi")
                                        print(f"Video will be saved to: {output_file}")

                                        # Initialize VideoWriter for this camera (if recording is enabled)
                                        if is_recording_enabled:
                                            fps = 30  # Default FPS (can be adjusted if needed)
                                            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec for .avi files
                                            video_writers[camera_id] = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
                                            if not video_writers[camera_id].isOpened():
                                                print(f"Error: Could not open VideoWriter for camera {camera_id}.")
                                        else:
                                            print(f"Recording is disabled. VideoWriter for camera {camera_id} will not be initialized.")
                                    except Exception as e:
                                        print(f"Error initializing camera {camera_id}: {e}")

                                # Debug print: Number of cameras being processed
                                print(f"Number of cameras being processed: {len(datasets)}")
                            except Exception as e:
                                print(f"Error updating model or cameras: {e}")

            # Check if the model and datasets are initialized
            if model is None or datasets is None:
                print("Model or datasets not initialized. Skipping frame processing.")
                continue  # Skip processing until the model and datasets are set

            # Check if thresholds have been received
            if not thresholds_received:
                print("Waiting for thresholds from frontend...")
                continue  # Skip processing until thresholds are received

            # Process frames from all cameras
            for dataset in datasets:
                for path, img, im0s, vid_cap in dataset:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0

                    # Ensure im0s is a valid NumPy array
                    if im0s is None:
                        print("Error: im0s is None")
                        continue

                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    with torch.no_grad():
                        pred = model(img)[0]
                    with parameter_lock:
                        # Ensure conf_thres and iou_thres are not None
                        if conf_thres is None or iou_thres is None:
                            print("Error: Confidence or IOU threshold is None. Skipping frame.")
                            continue
                        pred = non_max_suppression(pred, conf_thres, iou_thres)
                        # print(f"Applied thresholds: Conf={conf_thres}, IOU={iou_thres}")  # Debug print

                    # Process detections
                    for i, det in enumerate(pred):
                        # Create a copy of im0s to draw bounding boxes
                        if isinstance(im0s, np.ndarray):  # Single camera
                            im0 = im0s.copy()
                        else:  # Multiple cameras or webcam
                            im0 = [x.copy() for x in im0s]

                        # If detections are found, draw bounding boxes
                        if det is not None and len(det):
                            # Rescale boxes to original image size
                            if isinstance(im0, np.ndarray):  # Single camera
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            else:  # Multiple cameras or webcam
                                for idx, frame in enumerate(im0):
                                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                            # Draw bounding boxes
                            if isinstance(im0, np.ndarray):  # Single camera
                                for *xyxy, conf, cls in det:
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            else:  # Multiple cameras or webcam
                                for idx, frame in enumerate(im0):
                                    for *xyxy, conf, cls in det:
                                        label = f'{names[int(cls)]} {conf:.2f}'
                                        plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)

                        # Send the processed frame (im0) to the frontend, regardless of detections
                        if not output_queue.full():
                            if isinstance(im0, np.ndarray):  # Single camera
                                camera_id = path  # Use the path as the camera ID
                                output_queue.put((im0, camera_id))  # Send im0 (processed frame)
                                # print(f"Frame sent to frontend for camera {camera_id}.")
                            else:  # Multiple cameras or webcam
                                for idx, frame in enumerate(im0):
                                    camera_id = path[idx] if isinstance(path, list) else path  # Handle batch processing
                                    output_queue.put((frame, camera_id))  # Send each frame individually
                                    # print(f"Frame sent to frontend for camera {camera_id}.")

                        # Save the processed frame to the corresponding video file (if recording is enabled)
                        if is_recording_enabled:
                            if isinstance(im0, np.ndarray):  # Single camera
                                if video_writers[camera_id].isOpened():
                                    video_writers[camera_id].write(im0)
                                    # print(f"Frame written to video file for camera {camera_id}.")
                                else:
                                    print(f"Error: VideoWriter for camera {camera_id} is not open.")
                            else:  # Multiple cameras or webcam
                                for idx, frame in enumerate(im0):
                                    if video_writers[camera_id].isOpened():
                                        video_writers[camera_id].write(frame)
                                        # print(f"Frame written to video file for camera {camera_id}.")
                                    else:
                                        print(f"Error: VideoWriter for camera {camera_id} is not open.")
                        # else:
                        #     print("Recording is disabled. Skipping frame save.")

    finally:
        # Clean up resources
        if datasets is not None:
            for dataset in datasets:
                if hasattr(dataset, 'cap'):
                    dataset.cap.release()  # Release the camera
        if device.type != 'cpu':
            torch.cuda.empty_cache()  # Clear GPU memory

        # Release all VideoWriter objects
        for camera_id, writer in video_writers.items():
            if writer.isOpened():
                writer.release()
                print(f"VideoWriter for camera {camera_id} released.")
            else:
                print(f"Error: VideoWriter for camera {camera_id} was not open.")
        print("Backend process stopped gracefully.")
