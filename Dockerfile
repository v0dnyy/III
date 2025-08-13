# 1. Используем официальный NVIDIA L4T PyTorch образ для JetPack 6.x с CUDA 12.x
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth2.0-py3

# 2. Устанавливаем рабочую директорию
WORKDIR /app

# 3. Копируем код и требования
COPY inference.py /app/
COPY req1.txt /app/
COPY yolo_s_v11_dropout_05_best.pt /app/

# 4. Устанавливаем системные зависимости, необходимые OpenCV и ffmpeg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 5. Обновляем pip и ставим Python-зависимости из req.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/req1.txt

# 6. Запуск: замените на вашу команду или передавайте аргументы при запуске контейнера
CMD ["python3", "inference.py --path_to_model_w=./yolo_s_v11_dropout_05_best.pt --from_cam --show_video"]
