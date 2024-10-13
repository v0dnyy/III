from ultralytics import YOLO

model = YOLO("yolo11m")

model.predict(r"C:/Users/vodnyy/Desktop/datasets/drone_dataset_yolo/dataset_txt/0094.jpg", save = True)
