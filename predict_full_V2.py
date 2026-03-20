import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segmentation_model import EyeSegmentationModel

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')
SEGMENTOR_PATH = os.path.join(BASE_DIR, 'eye_segmentation_final.pth')
INPUT_DIR = os.path.join(BASE_DIR, 'input_images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_results')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Цвета для визуализации (BGR)
COLOR_IRIS = (0, 255, 0)  # Зеленый для радужки
COLOR_PUPIL = (0, 0, 255)  # Красный для зрачка


# -----------------

def get_segmentation(model, img_crop):
    """Вспомогательная функция для получения маски из кропа"""
    h, w = img_crop.shape[:2]
    img_resized = cv2.resize(img_crop, (256, 256))
    input_tensor = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = torch.tensor(input_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output[0], dim=0).cpu().numpy().astype(np.uint8)

    # Возвращаем маску, растянутую до исходного размера кропа
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def process_pipeline():
    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR); return
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("Загрузка моделей...")
    detector = YOLO(DETECTOR_PATH)
    segmentor = EyeSegmentationModel(n_classes=3).to(DEVICE)
    segmentor.load_state_dict(torch.load(SEGMENTOR_PATH, map_location=DEVICE))
    segmentor.eval()

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print(f"Найдено изображений: {len(image_files)}")

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        full_img = cv2.imread(img_path)
        if full_img is None: continue

        h_orig, w_orig = full_img.shape[:2]
        display_img = full_img.copy()
        black_mask = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

        # 1. Пробуем найти глаза через YOLO
        results = detector.predict(full_img, conf=0.01, iou=0.3, agnostic_nms=True, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            print(f"[{img_name}] Найдено глаз: {len(boxes)}")
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Квадратный кроп
                bw, bh = x2 - x1, y2 - y1
                side = max(bw, bh)
                cx, cy = x1 + bw // 2, y1 + bh // 2
                nx1, ny1 = max(0, cx - side // 2), max(0, cy - side // 2)
                nx2, ny2 = min(w_orig, nx1 + side), min(h_orig, ny1 + side)

                eye_crop = full_img[ny1:ny2, nx1:nx2]
                mask = get_segmentation(segmentor, eye_crop)

                # Рисуем на маске и на фото
                temp_mask_color = np.zeros_like(eye_crop)
                temp_mask_color[mask == 1] = COLOR_IRIS
                temp_mask_color[mask == 2] = COLOR_PUPIL

                # Накладываем на итоговые изображения
                display_img[ny1:ny2, nx1:nx2] = cv2.addWeighted(display_img[ny1:ny2, nx1:nx2], 1.0, temp_mask_color,
                                                                0.5, 0)
                black_mask[ny1:ny2, nx1:nx2] = temp_mask_color
                cv2.rectangle(display_img, (nx1, ny1), (nx2, ny2), (255, 0, 0), 1)
        else:
            # 2. РЕЗЕРВНЫЙ ВАРИАНТ: YOLO не нашла глаз, сегментируем все фото
            print(f"[{img_name}] Глаза не найдены. Запуск сегментации всего фото...")
            mask = get_segmentation(segmentor, full_img)

            temp_mask_color = np.zeros_like(full_img)
            temp_mask_color[mask == 1] = COLOR_IRIS
            temp_mask_color[mask == 2] = COLOR_PUPIL

            display_img = cv2.addWeighted(full_img, 1.0, temp_mask_color, 0.5, 0)
            black_mask = temp_mask_color

        # Сохранение результатов
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"result_{img_name}"), display_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_only_{img_name}"), black_mask)
        print(f"Обработано: {img_name}")


if __name__ == "__main__":
    process_pipeline()
