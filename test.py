from ultralytics import YOLO

# Load a model
model = YOLO("/home/recomputer/Desktop/yolov8n_benchmark_coral/640_yolov8n_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

# Run Prediction
model.predict("/home/recomputer/Desktop/yolov8n_benchmark_coral/detection0.mp4",show=True,imgsz=640,conf =0.25)
