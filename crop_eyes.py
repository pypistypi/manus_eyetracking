import cv2
import os
from ultralytics import YOLO

# --- НАСТРОЙКИ ---
MODEL_PATH = 'runs/detect/train/weights/best.pt'
INPUT_DIR = 'raw_images'
OUTPUT_DIR = 'cropped_dataset'
TARGET_SIZE = (256, 256)
CONFIDENCE = 0.01  # Порог уверенности
IOU_THRESHOLD = 0.3  # Порог для удаления дубликатов (NMS)


# -----------------

def process_images():
    model = YOLO(MODEL_PATH)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.bmp', '.jpg', '.png'))]

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # agnostic_nms=True заставляет YOLO выбирать только ОДНУ лучшую рамку на одном месте,
        # даже если она претендует на разные классы (левый/правый)
        results = model(img, conf=CONFIDENCE, iou=IOU_THRESHOLD, agnostic_nms=True, verbose=False)

        found_count = len(results[0].boxes)
        print(f"Файл {img_name}: Найдено глаз: {found_count}")

        for i, box in enumerate(results[0].boxes):
            # 1. Получаем координаты
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 2. Получаем имя класса (ВАЖНО для сохранения!)
            label_id = int(box.cls[0])
            label_name = model.names[label_id]  # left_eye или right_eye

            # 3. Приведение к квадрату без искажений
            w, h = x2 - x1, y2 - y1
            center_x, center_y = x1 + w // 2, y1 + h // 2
            side = max(w, h)

            nx1 = max(0, center_x - side // 2)
            ny1 = max(0, center_y - side // 2)
            nx2 = min(img.shape[1], nx1 + side)
            ny2 = min(img.shape[0], ny1 + side)

            eye_crop = img[ny1:ny2, nx1:nx2]
            if eye_crop.size == 0: continue

            # 4. Resize
            eye_resized = cv2.resize(eye_crop, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

            # 5. Сохранение с уникальным именем
            base_name = os.path.splitext(img_name)[0]
            # Добавляем индекс i, чтобы если нашлось 2 левых глаза, они не перезаписывали друг друга
            save_name = f"{base_name}_{label_name}_{i}.png"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            cv2.imwrite(save_path, eye_resized)
            print(f"   Успешно сохранен: {save_name}")


if __name__ == "__main__":
    process_images()
