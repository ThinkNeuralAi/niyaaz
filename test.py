from ultralytics import YOLO

model = YOLO("/home/ubuntu/Niyaaz_biryani/niyaaz/models/best.pt")
# model = YOLO("yolov8n.pt")
# model = YOLO("yolo11n.pt")

print("Class count:", model.model.nc)
print("Class names:", model.model.names)
