import cv2
from ultralytics import YOLO
import random
import os

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


def detect_dir_files(model_w, path_to_dir):
    model = YOLO(model_w)
    for filename in os.listdir(path_to_dir):
        file_path = os.path.join(path_to_dir, filename)
        model.predict(file_path, save = True, verbose=False, save_txt = True)
    

def model_validation(model_w, path_to_data):
    model = YOLO(model_w)
    metrics = model.val(data = path_to_data)
    print(model.names)
    print(metrics.box.map, metrics.box.map50, metrics.box.map75, metrics.box.maps)



def process_video_with_detect(model_w, input_video_path, from_cam = False, show_video=True, save_video=False,
                              output_video_path="output_video.mp4"):
    model = YOLO(model_w)
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

        if save_video:
            out.write(frame)

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
    detect_dir_files(r"III\yolo_s_v11_dropout_05_best.pt", r"C:\Users\vodnyy\Desktop\work_III\datasets\sample")
    process_video_with_detect(r"III\yolo_s_v11_dropout_05_best.pt", r"C:\Users\vodnyy\Videos\2024-10-18 20-12-35.mp4", from_cam = False, show_video=True, save_video=False,
                          output_video_path="yolo_n_v11_dropout_output_video.mp4")


if __name__ == '__main__':
    main()