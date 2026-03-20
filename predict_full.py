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

COLOR_IRIS = (0, 255, 0)  # Зеленый
COLOR_PUPIL = (0, 0, 255)  # Красный
COLOR_GLINT = (255, 255, 255)  # Белый для отблесков


# -----------------

def starburst_refinement(image_gray, center, num_rays=32, max_radius=100):
    """
    Реализация алгоритма Starburst: пускаем лучи из центра и ищем градиент.
    """
    points = []
    cx, cy = center

    # Проходим по кругу (360 градусов)
    for angle in np.linspace(0, 2 * np.pi, num_rays):
        dx = np.cos(angle)
        dy = np.sin(angle)

        last_val = None
        # Двигаемся вдоль луча
        for r in range(5, max_radius):
            px = int(cx + r * dx)
            py = int(cy + r * dy)

            if px < 0 or px >= image_gray.shape[1] or py < 0 or py >= image_gray.shape[0]:
                break

            val = image_gray[py, px]
            if last_val is not None:
                # Ищем резкий переход от темного (зрачок) к светлому (радужка)
                diff = int(val) - int(last_val)
                if diff > 15:  # Порог градиента
                    points.append((px, py))
                    break
            last_val = val

    return points


def detect_glints(image_gray, pupil_mask):
    """
    Детекция отблесков (glints): очень яркие точки внутри или около зрачка.
    """
    # Ищем самые яркие пиксели (близкие к 255)
    _, glints = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)
    # Оставляем только те, что в зоне глаза (используем маску)
    glints = cv2.bitwise_and(glints, glints, mask=cv2.dilate(pupil_mask, None, iterations=5))
    return glints


def process_pipeline():
    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR); return
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    detector = YOLO(DETECTOR_PATH)
    segmentor = EyeSegmentationModel(n_classes=3).to(DEVICE)
    segmentor.load_state_dict(torch.load(SEGMENTOR_PATH, map_location=DEVICE))
    segmentor.eval()

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    for img_name in image_files:
        full_img = cv2.imread(os.path.join(INPUT_DIR, img_name))
        if full_img is None: continue

        gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        display_img = full_img.copy()
        h_orig, w_orig = full_img.shape[:2]
        black_mask = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

        # 1. YOLO Детекция
        results = detector.predict(full_img, conf=0.01, verbose=False)

        # Если YOLO не нашла, работаем со всем фото как с одним "глазом"
        crops = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crops.append((x1, y1, x2, y2))
        else:
            crops.append((0, 0, w_orig, h_orig))

        for (x1, y1, x2, y2) in crops:
            # Кроп и сегментация нейросетью
            eye_crop = full_img[y1:y2, x1:x2]
            if eye_crop.size == 0: continue

            # Получаем маску от U-Net
            img_input = cv2.resize(eye_crop, (256, 256))
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
            with torch.no_grad():
                out = segmentor(torch.tensor(img_input).unsqueeze(0).to(DEVICE))
                mask_256 = torch.argmax(out[0], dim=0).cpu().numpy().astype(np.uint8)

            mask = cv2.resize(mask_256, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            # --- STARBURST ---
            # Находим центр маски зрачка (класс 2)
            pupil_pixels = np.where(mask == 2)
            if len(pupil_pixels[0]) > 0:
                mcy, mcx = np.mean(pupil_pixels[0]), np.mean(pupil_pixels[1])
                # Пускаем лучи Starburst для уточнения границы
                edge_points = starburst_refinement(gray[y1:y2, x1:x2], (int(mcx), int(mcy)))

                # Рисуем лучи Starburst на итоговом фото
                for pt in edge_points:
                    cv2.circle(display_img, (x1 + pt[0], y1 + pt[1]), 1, (255, 255, 0), -1)

                # Если точек достаточно, строим эллипс (Ellipse Fitting)
                if len(edge_points) >= 5:
                    ellipse = cv2.fitEllipse(np.array(edge_points))
                    ellipse = ((ellipse[0][0] + x1, ellipse[0][1] + y1), ellipse[1], ellipse[2])
                    cv2.ellipse(display_img, ellipse, (0, 255, 255), 2)  # Желтый эллипс

            # --- GLINTS (ОТБЛЕСКИ) ---
            glints = detect_glints(gray[y1:y2, x1:x2], (mask == 2).astype(np.uint8))
            display_img[y1:y2, x1:x2][glints > 0] = COLOR_GLINT
            black_mask[y1:y2, x1:x2][glints > 0] = COLOR_GLINT

            # Отрисовка масок нейросети
            temp_mask = np.zeros_like(eye_crop)
            temp_mask[mask == 1] = COLOR_IRIS
            temp_mask[mask == 2] = COLOR_PUPIL
            display_img[y1:y2, x1:x2] = cv2.addWeighted(display_img[y1:y2, x1:x2], 1.0, temp_mask, 0.3, 0)
            black_mask[y1:y2, x1:x2][mask == 1] = COLOR_IRIS
            black_mask[y1:y2, x1:x2][mask == 2] = COLOR_PUPIL

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"result_{img_name}"), display_img)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"mask_only_{img_name}"), black_mask)
        print(f"Готово: {img_name}")


if __name__ == "__main__":
    process_pipeline()
