from ultralytics import YOLO

# For YOLO (Ultralytics)
model = YOLO('/home/ubuntu/Niyaaz_biryani/niyaaz/models/yolo11n.pt')
model.export(format='onnx', dynamic=True, simplify=True)

# For Custom Model (Ultralytics)
model = YOLO('/home/ubuntu/Niyaaz_biryani/niyaaz/models/best.pt')
model.export(format='onnx', dynamic=True, simplify=True)

print("ONNX model exported successfully!")



# Build TensorRT engine from ONNX
#For YOLO model
trtexec --onnx=/home/ubuntu/Niyaaz_biryani/niyaaz/models/yolo11n.onnx \
--saveEngine=/home/ubuntu/Niyaaz_biryani/niyaaz/models/yolo11n.engine \
--fp16 \
--minShapes=images:1x3x640x640 \
--optShapes=images:4x3x640x640 \
--maxShapes=images:8x3x640x640

#For Custom Model
trtexec --onnx=/home/ubuntu/Niyaaz_biryani/niyaaz/models/best.onnx \
--saveEngine=/home/ubuntu/Niyaaz_biryani/niyaaz/models/best.engine \
--fp16 \
--minShapes=images:1x3x640x640 \
--optShapes=images:4x3x640x640 \
--maxShapes=images:8x3x640x640