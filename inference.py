import cv2
from ultralytics import YOLO
import random
import os
import json


def log_detected_objects(model, results):
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    conf = results[0].boxes.conf.cpu().numpy().astype(float)
    res = []
    for box, clss, conf in zip(boxes, classes, conf):
        res.append({
            "class": model.names[int(clss)],
            "confidence": conf.item(),
            "bounding_box": {
                "x1": box[0].item(),
                "y1": box[1].item(),
                "x2": box[2].item(),
                "y2": box[3].item()
            }
        })
    return res


def draw_bounding_boxes(model, frame, results):
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, clss in zip(boxes, classes):
        random.seed(int(clss) + 8)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
        cv2.putText(
            frame,
            f"{model.model.names[clss]}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 255, 50),
            2,
        )
    return frame


def detect_dir_files(path_to_model_w, path_to_dir):
    model = YOLO(path_to_model_w)
    for filename in os.listdir(path_to_dir):
        file_path = os.path.join(path_to_dir, filename)
        result = model.predict(file_path, save=True, verbose=False)
        with open(filename[:len(filename) - 4]+"_detected_objects.json", 'w+', encoding='utf-8') as file:
            json.dump(log_detected_objects(model, result), file, ensure_ascii=False, indent=4)


def model_validation(path_to_model_w, path_to_data):
    model = YOLO(path_to_model_w)
    metrics = model.val(data=path_to_data)
    print(model.names)
    print(metrics.box.map, metrics.box.map50, metrics.box.map75, metrics.box.maps)


def process_video_with_detect(path_to_model_w, input_video_path, from_cam=False, show_video=True, save_video=False,
                              save_logs=False,
                              output_video_path="output_video.mp4"):
    model = YOLO(path_to_model_w)
    result_json = []
    model.fuse()
    # Open the input video file
    if from_cam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get input video frame rate and dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output video writer
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, iou=0.4, conf=0.6, imgsz=640, verbose=False)

        if results[0].boxes != None:
            draw_bounding_boxes(model, frame, results)
            result_json.append(log_detected_objects(model, results))

        if save_video:
            out.write(frame)

        if save_logs:
            with open('detected_objects.json', 'w+', encoding='utf-8') as file:
                json.dump(result_json, file, ensure_ascii=False, indent=4)

        if show_video:
            frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6)
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_video:
        out.release()

    cv2.destroyAllWindows()


def main():
    detect_dir_files(r"C:\Users\vodnyy\III\III\yolo_s_v11_dropout_05_best.pt", r"C:\Users\vodnyy\Desktop\work_III\datasets\sample")
    process_video_with_detect(r"III\yolo_s_v11_dropout_05_best.pt",
                              r"III\demo.mp4",
                              from_cam=False, show_video=True,
                              save_video=True, save_logs=False,
                              output_video_path="demo_detected.mp4")


if __name__ == '__main__':
    main()
