from ultralytics import YOLO

# 1. Загружаем предобученную модель "Nano" (самая быстрая)
model = YOLO('yolov8n.pt')

# 2. Запускаем дообучение на ваших глазах
# Вам нужно будет создать файл data.yaml (я помогу с этим ниже)
results = model.train(
    data='eye_data.yaml',
    epochs=50,
    imgsz=640,
    plots=True
)

print("Обучение завершено! Модель сохранена в папку runs/detect/train/weights/best.pt")
