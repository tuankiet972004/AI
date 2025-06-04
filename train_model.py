from ultralytics import YOLO

model = YOLO('model/yolov8n.pt')
results = model.train(data = 'model/mydata.yaml', epochs = 50)