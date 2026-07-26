import os
import sys
import shutil
import time
import pickle
import subprocess
from pathlib import Path
from io import StringIO
import numpy as np
import cv2
from typing import Optional, List, Dict

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


def _fix_cuda_path():
    if sys.platform != 'win32':
        return
    candidates = [
        Path(sys.prefix) / 'Lib' / 'site-packages',
    ]
    env_path = os.environ.get('CONDA_PREFIX')
    if env_path:
        candidates.append(Path(env_path) / 'Lib' / 'site-packages')
    seen = set()
    extra = []
    for base in candidates:
        if not base.exists() or base in seen:
            continue
        seen.add(base)
        for pkg in base.glob('nvidia*'):
            bin_dir = pkg / 'bin'
            if bin_dir.exists():
                extra.append(str(bin_dir))
    if extra:
        os.environ['PATH'] = os.pathsep.join(extra) + os.pathsep + os.environ.get('PATH', '')


_fix_cuda_path()

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("❌ InsightFace не найден. Установите: pip install insightface")
    sys.exit(1)

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import pairwise_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# === НАСТРОЙКИ ===
BASE_DIR = Path(r"C:\Foto")
BASE_FOLDER = BASE_DIR / "Baza"
GUYS_FOLDER = BASE_DIR / "Parni"
OUTPUT_MATCHES = BASE_DIR / "Sovpadenia"
OUTPUT_MATCHES.mkdir(exist_ok=True)
GUYS_FOLDER.mkdir(exist_ok=True)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.webp', '.WEBP'}
CURRENT_INSIGHT_MODEL = 'buffalo_l'
GENDER_CONFIDENCE_THRESHOLD = 0.7
MALE_GENDER_ID = 0
MATCH_THRESHOLD = 0.65
CACHE_FILE = BASE_DIR / ".embeddings_cache.pkl"
CPU_DET_SIZE = (320, 320)
GPU_DET_SIZE = (640, 640)


def get_image_paths(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"⚠️ Папка не найдена: {folder}")
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def load_image(path: Path) -> Optional[np.ndarray]:
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_best_python() -> Optional[str]:
    candidates = [
        Path(sys.executable),
        Path(r"C:\Users\vyach\AppData\Local\Python\pythoncore-3.14-64\python.exe"),
        Path(r"C:\Users\vyach\AppData\Local\Programs\Python\Python314\python.exe"),
        Path(r"C:\Python314\python.exe"),
    ]
    for p in candidates:
        if p.exists():
            try:
                out = subprocess.check_output([str(p), "--version"], stderr=subprocess.STDOUT, text=True, timeout=10)
                ver = out.strip().lower()
                if "python 3." in ver:
                    return str(p)
            except Exception:
                continue
    return sys.executable


def init_face_analysis():
    providers = ['CPUExecutionProvider']
    det_size = CPU_DET_SIZE
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if 'CUDAExecutionProvider' in avail:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            det_size = GPU_DET_SIZE
            print("🟢 GPU (CUDA) доступен")
        else:
            print("🟡 GPU не найден, используем CPU")
            print(f"  det_size={det_size} для ускорения CPU")
    except ImportError:
        print("🟡 onnxruntime не найден, используем CPU")

    app = FaceAnalysis(name=CURRENT_INSIGHT_MODEL, root=os.path.join(os.path.expanduser('~'), '.insightface', 'models'), providers=providers)
    app.prepare(ctx_id=0 if providers[0] == 'CUDAExecutionProvider' else -1, det_size=det_size)
    print("✅ InsightFace инициализирован")
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


def safe_move(src: Path, dst_dir: Path, new_name: str) -> bool:
    dst = dst_dir / new_name
    if dst.exists():
        stem = Path(new_name).stem
        ext = Path(new_name).suffix
        counter = 1
        while dst.exists():
            new_name = f"{stem}_{counter}{ext}"
            dst = dst_dir / new_name
            counter += 1
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        print(f"  ❌ Ошибка перемещения {src.name}: {e}")
        return False


def cosine_distance(emb1, emb2) -> float:
    v1 = np.array(emb1, dtype=np.float32)
    v2 = np.array(emb2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1.0 - float(np.dot(v1, v2) / (norm1 * norm2))


def load_cache() -> Dict[str, Optional[np.ndarray]]:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict[str, Optional[np.ndarray]]):
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"⚠️ Не удалось сохранить кэш: {e}")


# === CPU-OPT: батч-обработка через нативный метод InsightFace ===

def batch_get_embeddings(app, images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    if hasattr(app, 'get'):
        return app.get(images)


def analyze_gender(app, cache: Dict):
    print("=" * 60)
    print("🔍 РЕЖИМ 1: Анализ пола (База → Парни)")
    print(f"📁 База: {BASE_FOLDER}")
    print(f"👦 Парни: {GUYS_FOLDER}")
    print("=" * 60)

    images = get_image_paths(BASE_FOLDER)
    print(f"Найдено файлов: {len(images)}")

    male_count = 0
    female_count = 0
    no_gender_count = 0
    no_face_count = 0
    error_count = 0
    to_process = []
    paths = []

    for idx, img in enumerate(images, 1):
        if idx % 100 == 0 or idx == len(images):
            print(f"  Прогресс: {idx}/{len(images)}")

        try:
            img_bgr = load_image(img)
            if img_bgr is None:
                no_face_count += 1
                continue
            to_process.append(img_bgr)
            paths.append((img, img_bgr.shape))
        except Exception as e:
            error_count += 1
            print(f"  ❌ Ошибка загрузки {img.name}: {e}")

    try:
        batch_results = app.get(to_process)
    except Exception:
        batch_results = [app.get(img)[0] if app.get(img) else None for img in to_process]

    for img_path, faces in zip(paths, batch_results):
        if not faces:
            no_face_count += 1
            continue
        faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        face = faces[0]
        gender = get_gender(face)
        if gender == MALE_GENDER_ID:
            safe_move(img_path[0], GUYS_FOLDER, img_path[0].name)
            male_count += 1
        elif gender == 1:
            female_count += 1
        else:
            no_gender_count += 1

    print("\n" + "=" * 60)
    print("🎉 ГОТОВО!")
    print(f"👦 Мужские лица перенесено: {male_count}")
    print(f"👩 Женские лица осталось: {female_count}")
    print(f"❓ Пол не определён: {no_gender_count}")
    print(f"😐 Без лица: {no_face_count}")
    print(f"⚠️ Ошибки: {error_count}")
    print("=" * 60)


# === РЕЖИМ 2 и 3: ПОИСК ДУБЛЕЙ ===

def find_duplicates_in_folder(source_folder: Path, app, cache: Dict, output_folder: Path):
    print("=" * 60)
    print(f"🔍 ПОИСК ДУБЛЕЙ: {source_folder}")
    print(f"📂 Цель: {output_folder}")
    print("=" * 60)

    images = get_image_paths(source_folder)
    print(f"Найдено файлов: {len(images)}")

    # Загружаем изображения батчем
    imgs_bgr = []
    paths = []
    for idx, img in enumerate(images, 1):
        if idx % 100 == 0 or idx == len(images):
            print(f"  Загрузка: {idx}/{len(images)}")
        bgr = load_image(img)
        if bgr is not None:
            imgs_bgr.append(bgr)
            paths.append(img)

    print(f"Загружено изображений: {len(imgs_bgr)}")
    print(f"Извлечение эмбеддингов...")

    try:
        batch_results = app.get(imgs_bgr)
        if batch_results is None:
            batch_results = []
    except Exception:
        batch_results = []
        for img in imgs_bgr:
            try:
                result = app.get(img)
                if result:
                    batch_results.append(result)
                else:
                    batch_results.append(None)
            except Exception:
                batch_results.append(None)

    valid_images = []
    embeddings = []
    for img_path, faces in zip(paths, batch_results):
        if faces:
            faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            emb = faces[0].embedding
            cache[img_path.name] = emb
            valid_images.append(img_path)
            embeddings.append(np.array(emb, dtype=np.float32))
        else:
            cache[img_path.name] = None

    save_cache(cache)
    print(f"Валидных лиц: {len(valid_images)}")

    if len(valid_images) < 2:
        print("Недостаточно файлов для поиска дублей.")
        return

    used = [False] * len(valid_images)
    groups_found = 0
    total_moved = 0

    if SKLEARN_AVAILABLE and len(embeddings) > 50:
        print("🚀 Используем sklearn DBSCAN для группировки...")
        emb_matrix = np.vstack(embeddings)
        dists = pairwise_distances(emb_matrix, metric='cosine')
        clustering = DBSCAN(eps=MATCH_THRESHOLD, min_samples=2, metric='precomputed').fit(dists)
        labels = clustering.labels_

        groups: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            groups.setdefault(int(label), []).append(i)

        for label, indices in groups.items():
            if len(indices) < 2:
                continue
            group_files = sorted([valid_images[i] for i in indices], key=lambda p: p.name)
            representative = group_files[0].stem
            target_dir = output_folder / representative
            target_dir.mkdir(exist_ok=True)

            moved = 0
            for f in group_files:
                if safe_move(f, target_dir, f.name):
                    moved += 1

            groups_found += 1
            total_moved += moved
            print(f"  📁 '{representative}': {len(group_files)} файлов, перемещено {moved}")
    else:
        print("🔍 Простой перебор...")
        for i in range(len(valid_images)):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(valid_images)):
                if used[j]:
                    continue
                dist = cosine_distance(embeddings[i], embeddings[j])
                if dist <= MATCH_THRESHOLD:
                    group.append(j)

            if len(group) >= 2:
                group_files = sorted([valid_images[k] for k in group], key=lambda p: p.name)
                representative = group_files[0].stem
                target_dir = output_folder / representative
                target_dir.mkdir(exist_ok=True)

                moved = 0
                for f in group_files:
                    if safe_move(f, target_dir, f.name):
                        moved += 1
                        for idx, vi in enumerate(valid_images):
                            if vi == f:
                                used[idx] = True
                                break

                groups_found += 1
                total_moved += moved
                print(f"  📁 '{representative}': {len(group_files)} файлов, перемещено {moved}")

    print("\n" + "=" * 60)
    print("🎉 ГОТОВО!")
    print(f"📁 Групп дублей найдено: {groups_found}")
    print(f"📦 Файлов перемещено: {total_moved}")
    print("=" * 60)


# === ЗАПУСК ПО УМОЛЧАНИЮ (ОБХОД МЕНЮ) ===
# ─────────────────────────────────────
# Если скрипт запустить с аргументом 1 / 2 / 3 в консоли, режим выберется сам.
# Если запустить двойным кликом / без аргументов — покажется меню (input).

def run_default_mode():
    app = None
    cache = load_cache()

    if len(sys.argv) > 1:
        mode = sys.argv[1].strip()
    else:
        mode = None

    if mode == "1":
        if app is None:
            app = init_face_analysis()
        analyze_gender(app, cache)
    elif mode == "2":
        if app is None:
            app = init_face_analysis()
        find_duplicates_in_folder(GUYS_FOLDER, app, cache, OUTPUT_MATCHES)
    elif mode == "3":
        if app is None:
            app = init_face_analysis()
        find_duplicates_in_folder(BASE_FOLDER, app, cache, OUTPUT_MATCHES)
    else:
        main_menu()


# === ГЛАВНОЕ МЕНЮ ===

def main_menu():
    app = None
    cache = load_cache()

    while True:
        print("\n" + "=" * 60)
        print("📸 ОБРАБОТКА ФОТОГРАФИЙ")
        print("=" * 60)
        print("1. Анализ пола (База → Парни)")
        print("2. Дубли в Парни → Совпадения")
        print("3. Дубли в База → Совпадения")
        print("4. Выход")
        print("=" * 60)

        try:
            choice = input("Выберите режим (1-4): ").strip()
        except EOFError:
            print("\n👋 Выход.")
            break

        if choice == '1':
            if app is None:
                app = init_face_analysis()
            analyze_gender(app, cache)
        elif choice == '2':
            if app is None:
                app = init_face_analysis()
            find_duplicates_in_folder(GUYS_FOLDER, app, cache, OUTPUT_MATCHES)
        elif choice == '3':
            if app is None:
                app = init_face_analysis()
            find_duplicates_in_folder(BASE_FOLDER, app, cache, OUTPUT_MATCHES)
        elif choice == '4':
            print("👋 Выход.")
            break
        else:
            print("⚠️ Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    run_default_mode()
