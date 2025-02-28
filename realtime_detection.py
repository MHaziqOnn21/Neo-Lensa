import time
import cv2
import torch
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
import os
from datetime import datetime
import gc
import psutil

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import threading
from threading import Lock

def run_inference(input_queue, output_queue, camera_id_map):
    # Initialize settings
    camera_ids = None
    weights = None
    imgsz = 640
    conf_thres = None
    iou_thres = None
    device = select_device('')
    half = device.type != 'cpu'

    # Print the device being used
    if device.type == 'cpu':
        print("Using CPU for inference.")
    else:
        gpu_name = torch.cuda.get_device_name(device.index)
        print(f"Using GPU ({gpu_name}) for inference.")

    # Lock for thread-safe parameter updates
    parameter_lock = threading.Lock()

    # Initialize model and datasets
    model = None
    datasets = None

    # Dictionary to store VideoWriter objects for each camera
    video_writers = {}

    # Flag to indicate if thresholds have been received
    thresholds_received = False

    # Flag to indicate if recording is enabled
    is_recording_enabled = True

    try:
        while True:
            print("Backend process is running...")

            # Check for new settings or stop signal from the frontend
            if not input_queue.empty():
                new_settings = input_queue.get()
                if new_settings == "stop":
                    print("Stop signal received. Exiting...")
                    break
                elif isinstance(new_settings, tuple):
                    if new_settings[0] == "update_thresholds":
                        with parameter_lock:
                            _, conf_thres, iou_thres = new_settings
                            thresholds_received = True
                            print(f"Thresholds received from frontend: Conf={conf_thres}, IOU={iou_thres}")
                    elif new_settings[0] == "update_model":
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
                                                if isinstance(im0s, np.ndarray):
                                                    frame_height, frame_width = im0s.shape[:2]
                                                else:
                                                    frame_height, frame_width = im0s[0].shape[:2]
                                                break

                                        # Map camera ID to Cam1 or Cam2
                                        cam_id_display = camera_id_map.get(camera_id, f"Cam{camera_id}")

                                        # Create directory structure
                                        now = datetime.now()
                                        month_year = now.strftime("%b_%y")
                                        date = now.strftime("%d-%b-%Y")
                                        time_str = now.strftime("%H.%M.%S")
                                        save_dir = os.path.join("RealtimeArchive", month_year, date)
                                        os.makedirs(save_dir, exist_ok=True)

                                        # Generate output file path
                                        output_file = os.path.join(save_dir, f"{cam_id_display}_{time_str}.avi")
                                        print(f"Video will be saved to: {output_file}")

                                        # Initialize VideoWriter for this camera (if recording is enabled)
                                        if is_recording_enabled:
                                            fps = 30
                                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                            video_writers[camera_id] = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
                                            if not video_writers[camera_id].isOpened():
                                                print(f"Error: Could not open VideoWriter for camera {camera_id}.")
                                        else:
                                            print(f"Recording is disabled. VideoWriter for camera {camera_id} will not be initialized.")
                                    except Exception as e:
                                        print(f"Error initializing camera {camera_id}: {e}")

                                print(f"Number of cameras being processed: {len(datasets)}")
                            except Exception as e:
                                print(f"Error updating model or cameras: {e}")

            # Check if the model and datasets are initialized
            if model is None or datasets is None:
                print("Model or datasets not initialized. Skipping frame processing.")
                continue

            # Check if thresholds have been received
            if not thresholds_received:
                print("Waiting for thresholds from frontend...")
                continue

            # Process frames from all cameras
            for dataset in datasets:
                for path, img, im0s, vid_cap in dataset:
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0

                    if im0s is None:
                        print("Error: im0s is None")
                        continue

                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    with torch.no_grad():
                        pred = model(img)[0]
                    with parameter_lock:
                        if conf_thres is None or iou_thres is None:
                            print("Error: Confidence or IOU threshold is None. Skipping frame.")
                            continue
                        pred = non_max_suppression(pred, conf_thres, iou_thres)

                    # Process detections
                    for i, det in enumerate(pred):
                        if isinstance(im0s, np.ndarray):
                            im0 = im0s.copy()
                        else:
                            im0 = [x.copy() for x in im0s]

                        if det is not None and len(det):
                            if isinstance(im0, np.ndarray):
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            else:
                                for idx, frame in enumerate(im0):
                                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                            if isinstance(im0, np.ndarray):
                                for *xyxy, conf, cls in det:
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            else:
                                for idx, frame in enumerate(im0):
                                    for *xyxy, conf, cls in det:
                                        label = f'{names[int(cls)]} {conf:.2f}'
                                        plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)

                        # Send the processed frame to the frontend
                        if not output_queue.full():
                            if isinstance(im0, np.ndarray):
                                camera_id = path
                                output_queue.put((im0, camera_id))
                            else:
                                for idx, frame in enumerate(im0):
                                    camera_id = path[idx] if isinstance(path, list) else path
                                    output_queue.put((frame, camera_id))

                        # Save the processed frame to the corresponding video file
                        if is_recording_enabled:
                            if isinstance(im0, np.ndarray):
                                if video_writers[camera_id].isOpened():
                                    video_writers[camera_id].write(im0)
                                else:
                                    print(f"Error: VideoWriter for camera {camera_id} is not open.")
                            else:
                                for idx, frame in enumerate(im0):
                                    if video_writers[camera_id].isOpened():
                                        video_writers[camera_id].write(frame)
                                    else:
                                        print(f"Error: VideoWriter for camera {camera_id} is not open.")

                    # Clear memory
                    del img, pred, det
                    torch.cuda.empty_cache()
                    gc.collect()

    finally:
        # Clean up resources
        if datasets is not None:
            for dataset in datasets:
                if hasattr(dataset, 'cap'):
                    dataset.cap.release()
        if device.type != 'cpu':
            torch.cuda.empty_cache()

        # Release all VideoWriter objects
        for camera_id, writer in video_writers.items():
            if writer.isOpened():
                writer.release()
                print(f"VideoWriter for camera {camera_id} released.")
            else:
                print(f"Error: VideoWriter for camera {camera_id} was not open.")
        print("Backend process stopped gracefully.")

