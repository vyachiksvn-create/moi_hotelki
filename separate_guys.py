import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
from typing import List, Optional

warnings = None
try:
    import warnings as w
    warnings = w
    w.filterwarnings("ignore")
except Exception:
    pass

try:
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("❌ InsightFace не найден. Установите: pip install insightface")
    sys.exit(1)

# === НАСТРОЙКИ ===
BASE_DIR = Path(r"C:\Users\vyach\Desktop\Новая папка")
BASE_FOLDER = BASE_DIR / "База"
GUYS_FOLDER = BASE_DIR / "Парни"
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.webp', '.WEBP'}

GUYS_FOLDER.mkdir(exist_ok=True)

GENDER_CONFIDENCE_THRESHOLD = 0.7
MALE_GENDER_ID = 0


def get_image_paths(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"⚠️ Папка не найдена: {folder}")
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def load_image(path: Path) -> Optional[np.ndarray]:
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def init_face_analysis():
    providers = ['CPUExecutionProvider']
    try:
        import onnxruntime as ort
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("🟢 GPU (CUDA) доступен")
        else:
            print("🟡 GPU не найден, используем CPU")
    except ImportError:
        print("🟡 onnxruntime не найден, используем CPU")

    app = FaceAnalysis(name='buffalo_l', root=os.path.join(os.path.expanduser('~'), '.insightface', 'models'))
    app.prepare(ctx_id=0 if providers[0] == 'CUDAExecutionProvider' else -1, det_size=(640, 640), providers=providers)
    print("✅ InsightFace initialized")
    return app


def get_gender(face) -> Optional[int]:
    if not hasattr(face, 'gender'):
        return None
    gender = face.gender
    if isinstance(gender, (int, np.integer)):
        return int(gender)
    if isinstance(gender, (list, tuple)) and len(gender) >= 1:
        g = int(gender[0])
        confidence = float(gender[1]) if len(gender) > 1 else 1.0
        if confidence < GENDER_CONFIDENCE_THRESHOLD:
            return None
        return g
    return None


def safe_move(src: Path, dst_dir: Path) -> Optional[Path]:
    dst = dst_dir / src.name
    if not dst.exists():
        try:
            shutil.move(str(src), str(dst))
            return dst
        except Exception as e:
            print(f"  ❌ Ошибка перемещения {src.name}: {e}")
            return None
    stem = src.stem
    ext = src.suffix
    counter = 1
    while dst.exists():
        dst = dst_dir / f"{stem}_{counter}{ext}"
        counter += 1
    try:
        shutil.move(str(src), str(dst))
        return dst
    except Exception as e:
        print(f"  ❌ Ошибка перемещения {src.name}: {e}")
        return None


def main():
    start_time = time.time()
    print("=" * 60)
    print("🔍 РАЗДЕЛЕНИЕ ПО ПОЛУ: База → Парни")
    print(f"📁 База: {BASE_FOLDER}")
    print(f"👦 Парни: {GUYS_FOLDER}")
    print("=" * 60)

    app = init_face_analysis()
    images = get_image_paths(BASE_FOLDER)
    print(f"\n📷 Найдено файлов: {len(images)}")

    male_count = 0
    female_count = 0
    no_gender_count = 0
    no_face_count = 0
    error_count = 0
    debug_log = []

    for idx, img in enumerate(images, 1):
        if idx % 100 == 0 or idx == len(images):
            print(f"  Обработка: {idx}/{len(images)}")

        try:
            img_bgr = load_image(img)
            if img_bgr is None:
                error_count += 1
                continue

            faces = app.get(img_bgr)
            if not faces:
                no_face_count += 1
                continue

            faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            best_face = faces[0]

            gender = get_gender(best_face)
            if gender == MALE_GENDER_ID:
                safe_move(img, GUYS_FOLDER)
                male_count += 1
            elif gender == 1:
                female_count += 1
            else:
                no_gender_count += 1

        except Exception as e:
            error_count += 1
            debug_log.append(f"❌ Ошибка {img.name}: {e}")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("🎉 ГОТОВО!")
    print(f"⏱️  Время: {total_time:.1f} сек.")
    print(f"👦 Мужские лица: {male_count}")
    print(f"👩 Женские лица: {female_count}")
    print(f"❓ Пол не определён: {no_gender_count}")
    print(f"😐 Без лица: {no_face_count}")
    print(f"⚠️ Ошибки: {error_count}")
    if debug_log:
        print("\nЛог ошибок:")
        for err in debug_log:
            print(f"  {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()