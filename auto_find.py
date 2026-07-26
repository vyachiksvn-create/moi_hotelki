import os
import sys
import shutil
import time
import re
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import warnings
import cv2
from typing import Optional, List, Tuple, Dict

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace не найден. Переключаюсь на резервный режим DeepFace.")

warnings.filterwarnings("ignore")

try:
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# === НАСТРОЙКИ ===
BASE_DIR = Path(r"C:\Users\vyach\Desktop\Новая папка")
BASE_FOLDER = BASE_DIR / "База"
NUMBERS_FOLDER = BASE_DIR / "Цифры"

OUTPUT_DUPLICATES_MULTI = BASE_DIR / "Новая папка_Дубли (2 и более фото)"
OUTPUT_DUPLICATES_MULTI.mkdir(exist_ok=True)

GUYS_FOLDER = BASE_DIR / "Парни"
GUYS_FOLDER.mkdir(exist_ok=True)

NEAR_DUPLICATES_FOLDER = BASE_DIR / "Sovpadenia"
NEAR_DUPLICATES_FOLDER.mkdir(exist_ok=True)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.webp', '.WEBP'}

THRESHOLDS_INSIGHT = {
    'arcface_r100_v1': 0.45,
    'buffalo_l': 0.55
}
MIN_AGREE_INSIGHT = 2
CURRENT_INSIGHT_MODEL = 'buffalo_l'
GENDER_CONFIDENCE_THRESHOLD = 0.7

if not INSIGHTFACE_AVAILABLE:
    from deepface import DeepFace
    MODELS_DF = ['Facenet512', 'ArcFace']
    THRESHOLDS_DF = {'Facenet512': 0.60, 'ArcFace': 0.50}
    MIN_AGREE_DF = 2
    DETECTOR_CHAIN = ['retinaface', 'mtcnn']
    print("⚠️ Используем резервный режим DeepFace (медленнее)")
else:
    print("🚀 Используем InsightFace (быстро и точно)")


def get_image_paths(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"⚠️ Папка не найдена: {folder}")
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]


def load_image(path: Path) -> Optional[np.ndarray]:
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def sort_faces_by_size(faces) -> list:
    faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
    return faces


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


def cosine_distance(emb1, emb2) -> float:
    v1 = np.array(emb1, dtype=np.float32)
    v2 = np.array(emb2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1.0 - float(np.dot(v1, v2) / (norm1 * norm2))


def extract_embeddings_insightface(img_bgr: np.ndarray, app: FaceAnalysis) -> Optional[np.ndarray]:
    faces = app.get(img_bgr)
    if not faces:
        return None
    faces = sort_faces_by_size(faces)
    return faces[0].embedding


def extract_embeddings_deepface(image_path: Path, cache: Dict[str, Optional[np.ndarray]], debug_log: list) -> Optional[np.ndarray]:
    if image_path.name in cache:
        return cache[image_path.name]

    tmpdir = tempfile.gettempdir()
    tmp_name = f"tmp_df_{image_path.stem.encode('utf-8').hex()}{image_path.suffix}"
    tmp_path = Path(tmpdir) / tmp_name
    embedding = None

    try:
        shutil.copy2(image_path, tmp_path)
        for model_name in MODELS_DF:
            for detector in DETECTOR_CHAIN:
                try:
                    result = DeepFace.represent(
                        str(tmp_path),
                        model_name=model_name,
                        detector_backend=detector,
                        enforce_detection=False,
                        align=True
                    )
                    if result and len(result) > 0:
                        emb = result[0].get("embedding")
                        if emb:
                            embedding = np.array(emb, dtype=np.float32)
                            cache[image_path.name] = embedding
                            break
                except Exception:
                    continue
            if embedding is not None:
                break
    except Exception as e:
        debug_log.append(f"❌ DF Error {image_path.name}: {e}")
        cache[image_path.name] = None
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    return embedding


def is_match(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> Tuple[bool, float]:
    if emb1 is None or emb2 is None:
        return False, 1.0
    dist = cosine_distance(emb1, emb2)
    threshold = THRESHOLDS_INSIGHT.get(CURRENT_INSIGHT_MODEL, 0.55) if INSIGHTFACE_AVAILABLE else THRESHOLDS_DF.get('Facenet512', 0.60)
    return dist <= threshold, dist


def has_latin(text: str) -> bool:
    return bool(re.search(r'[A-Za-z]', text))


def has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[\u0400-\u04FF]', text))


def normalize_name(name: str) -> str:
    text = Path(name).stem.lower()
    text = re.sub(r'[^a-zа-яё\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_fio_parts(text: str) -> Tuple[str, str, str]:
    text = normalize_name(text)
    parts = text.split()
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], parts[1], ''
    elif len(parts) == 1:
        return parts[0], '', ''
    return '', '', ''


def name_matches(a: str, b: str) -> Tuple[bool, bool]:
    fn1, sn1, pn1 = extract_fio_parts(a)
    fn2, sn2, pn2 = extract_fio_parts(b)
    exact_fio = bool(fn1 and sn1 and pn1 and fn1 == fn2 and sn1 == sn2 and pn1 == pn2)
    fi_only = bool(fn1 and sn1 and fn1 == fn2 and sn1 == sn2)
    return exact_fio, fi_only


def safe_move(src: Path, dst_dir: Path, new_name: str) -> Optional[Path]:
    dst = dst_dir / new_name
    if not dst.exists():
        try:
            shutil.move(str(src), str(dst))
            return dst
        except Exception as e:
            print(f"  ❌ Ошибка перемещения {src.name}: {e}")
            return None
    stem = Path(new_name).stem
    ext = Path(new_name).suffix
    counter = 1
    while dst.exists():
        new_name = f"{stem}_{counter}{ext}"
        dst = dst_dir / new_name
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
    print("🚀 ПОИСК ДУБЛЕЙ: БАЗА → ЦИФРЫ")
    print(f"📁 База: {BASE_FOLDER}")
    print(f"🔢 Цифры: {NUMBERS_FOLDER}")
    print(f"👦 Парни: {GUYS_FOLDER}")
    print("=" * 60)

    for f in OUTPUT_DUPLICATES_MULTI.iterdir():
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass

    debug_log = []
    numbers_embeddings_cache: Dict[str, Optional[np.ndarray]] = {}
    base_embeddings_cache: Dict[str, Optional[np.ndarray]] = {}

    # Инициализация InsightFace
    app = None
    if INSIGHTFACE_AVAILABLE:
        try:
            import onnxruntime as ort
            has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has_cuda else ['CPUExecutionProvider']
            if has_cuda:
                print("  🟢 GPU (CUDA) доступен")
            else:
                print("  🟡 GPU не найден, используем CPU")
        except ImportError:
            providers = ['CPUExecutionProvider']
            print("  🟡 onnxruntime не найден, используем CPU")

        try:
            app = FaceAnalysis(name=CURRENT_INSIGHT_MODEL, root=os.path.join(os.path.expanduser('~'), '.insightface', 'models'))
            app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1, det_size=(640, 640), providers=providers)
            print(f"✅ InsightFace initialized ({CURRENT_INSIGHT_MODEL})")
        except Exception as e:
            print(f"❌ Ошибка инициализации InsightFace: {e}")
            sys.exit(1)

    # ============================================================
    # ШАГ 1. Разделение по полу: Цифры → Парни
    # ============================================================
    print("\n[1/4] Разделение по полу (Цифры → Парни)...")
    male_count = 0
    female_count = 0
    no_gender_count = 0

    if INSIGHTFACE_AVAILABLE:
        numbers_images = get_image_paths(NUMBERS_FOLDER)
        print(f"  Найдено файлов в цифрах: {len(numbers_images)}")

        for idx, img in enumerate(numbers_images, 1):
            if idx % 10 == 0 or idx == len(numbers_images):
                print(f"  Обработка: {idx}/{len(numbers_images)}")

            img_bgr = load_image(img)
            if img_bgr is None:
                no_gender_count += 1
                continue

            try:
                faces = app.get(img_bgr)
                if not faces:
                    no_gender_count += 1
                    continue

                faces = sort_faces_by_size(faces)
                gender = get_gender(faces[0])

                if gender == 0:  # male
                    safe_move(img, GUYS_FOLDER, img.name)
                    male_count += 1
                elif gender == 1:  # female
                    female_count += 1
                    # Сохраняем эмбеддинг для последующего использования
                    numbers_embeddings_cache[img.name] = faces[0].embedding
                else:
                    no_gender_count += 1
                    # Сохраняем None, чтобы не обрабатывать повторно
                    numbers_embeddings_cache[img.name] = None
            except Exception as e:
                debug_log.append(f"❌ Ошибка обработки {img.name}: {e}")
                no_gender_count += 1
                numbers_embeddings_cache[img.name] = None

        print(f"  👦 Мужчин перенесено в Парни: {male_count}")
        print(f"  👩 Женщин осталось в Цифры: {female_count}")
        print(f"  ❓ Пол не определён: {no_gender_count}")
    else:
        print("  ⚠️ InsightFace недоступен — пропуск разделения по полу")

    # ============================================================
    # ШАГ 2. База — загрузка эмбеддингов
    # ============================================================
    print("\n[2/4] Загрузка базы...")
    base_images = get_image_paths(BASE_FOLDER)
    print(f"  Найдено файлов в базе: {len(base_images)}")

    base_faces = []
    for idx, img in enumerate(base_images, 1):
        if idx % 10 == 0 or idx == len(base_images):
            print(f"  База: {idx}/{len(base_images)}")

        if img.name in base_embeddings_cache:
            emb = base_embeddings_cache[img.name]
        else:
            if INSIGHTFACE_AVAILABLE and app is not None:
                img_bgr = load_image(img)
                if img_bgr is not None:
                    emb = extract_embeddings_insightface(img_bgr, app)
                else:
                    emb = None
            else:
                emb = extract_embeddings_deepface(img, base_embeddings_cache, debug_log)
            base_embeddings_cache[img.name] = emb

        if emb is not None:
            base_faces.append({
                'embedding': emb,
                'path': img,
                'name': img.stem,
                'ext': img.suffix,
            })
        else:
            debug_log.append(f"⚠️ База: {img.name} - лицо не найдено")

    print(f"  ✅ Распознано в базе: {len(base_faces)}")

    if not base_faces:
        print("❌ В базе нет распознанных лиц. Завершение.")
        if debug_log:
            print("\nЛог ошибок:")
            for err in debug_log:
                print(f"  {err}")
        return

    # ============================================================
    # ШАГ 3. Цифры (оставшиеся после фильтрации по полу)
    # ============================================================
    print("\n[3/4] Загрузка цифр...")
    numbers_images = get_image_paths(NUMBERS_FOLDER)
    print(f"  Найдено файлов в цифрах: {len(numbers_images)}")

    numbers_faces = []
    for idx, img in enumerate(numbers_images, 1):
        if idx % 10 == 0 or idx == len(numbers_images):
            print(f"  Цифры: {idx}/{len(numbers_images)}")

        if img.name in numbers_embeddings_cache:
            emb = numbers_embeddings_cache[img.name]
        else:
            if INSIGHTFACE_AVAILABLE and app is not None:
                img_bgr = load_image(img)
                if img_bgr is not None:
                    emb = extract_embeddings_insightface(img_bgr, app)
                else:
                    emb = None
            else:
                emb = extract_embeddings_deepface(img, numbers_embeddings_cache, debug_log)
            numbers_embeddings_cache[img.name] = emb

        if emb is not None:
            numbers_faces.append({
                'embedding': emb,
                'path': img,
                'name': img.stem,
                'ext': img.suffix,
                'matched': False,
            })
        else:
            debug_log.append(f"⚠️ Цифры: {img.name} - лицо не найдено")

    print(f"  ✅ Распознано в цифрах: {len(numbers_faces)}")

    # ============================================================
    # ШАГ 4. Поиск дублей
    # ============================================================
    print("\n[4/4] Поиск дублей...")

    dup_multi_count = 0
    base_no_match = 0

    for base_face in base_faces:
        base_name = base_face['name']
        base_ext = base_face['ext']
        base_emb = base_face['embedding']

        matches = []
        for i, num_face in enumerate(numbers_faces):
            if num_face['matched']:
                continue
            is_matched, dist = is_match(base_emb, num_face['embedding'])
            if is_matched:
                matches.append((i, dist))

        matches.sort(key=lambda x: x[1])

        if not matches:
            base_no_match += 1
            continue

        for idx, _, in matches:
            numbers_faces[idx]['matched'] = True

        dup_multi_count += 1
        target_dir = OUTPUT_DUPLICATES_MULTI

        base_new_name = f"{base_name}_1{base_ext}"
        base_new_path = safe_move(base_face['path'], target_dir, base_new_name)

        for i, (idx, dist) in enumerate(matches, start=2):
            num_face = numbers_faces[idx]
            num_new_name = f"{base_name}_{i}{num_face['ext']}"
            safe_move(num_face['path'], target_dir, num_new_name)
        print(f"  👥 {base_name}: {len(matches)} совпадений")

    # ============================================================
    # ШАГ 5. Пост-обработка: не-matched латиница -> Цифры
    # ============================================================
    print("\n[5/4] Пост-обработка: не-matched латиница -> Цифры...")
    moved_latin_to_numbers = 0
    for base_face in base_faces:
        img_path = base_face['path']
        if not img_path.exists():
            continue
        if has_latin(img_path.name):
            safe_move(img_path, NUMBERS_FOLDER, img_path.name)
            moved_latin_to_numbers += 1
    print(f"  🔤 Латинских файлов без дублей перенесено в Цифры: {moved_latin_to_numbers}")

    # ============================================================
    # ШАГ 6. ФИО/ФИ группировка внутри Baza -> Sovpadenia
    # ============================================================
    print("\n[6/4] ФИО/ФИ группировка внутри Baza...")
    remaining_base = [f for f in get_image_paths(BASE_FOLDER) if f.exists()]
    name_groups: Dict[str, List[Path]] = {}
    for img_path in remaining_base:
        norm = normalize_name(img_path.name)
        if not norm:
            continue
        fn, sn, pn = extract_fio_parts(norm)
        key_fio = f"{fn}|{sn}|{pn}"
        key_fi = f"{fn}|{sn}"
        name_groups.setdefault(key_fio, []).append(img_path)
        name_groups.setdefault(key_fi, []).append(img_path)

    seen_groups = set()
    moved_near_duplicates = 0
    for key, members in name_groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda p: p.name)
        group_key = tuple(sorted(str(p) for p in members))
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)

        names = [p.stem for p in members]
        has_doubt = len(set(extract_fio_parts(n)[2] for n in names)) > 1
        representative = members[0].stem

        if has_doubt:
            target_dir = NEAR_DUPLICATES_FOLDER
            logging.info(f"FIO near-duplicate (doubt) -> Sovpadenia root: {len(members)} files")
        else:
            target_dir = NEAR_DUPLICATES_FOLDER / representative
            logging.info(f"FIO duplicate -> Sovpadenia/{target_dir.name}: {len(members)} files")

        target_dir.mkdir(parents=True, exist_ok=True)
        for member in members:
            if safe_move(member, target_dir, member.name):
                moved_near_duplicates += 1

    print(f"  👥 Файлов по ФИО/ФИ перемещено в Sovpadenia: {moved_near_duplicates}")

    # ============================================================
    # ИТОГИ
    # ============================================================
    total_time = time.time() - start_time
    remaining_count = sum(1 for f in numbers_faces if not f['matched'])

    stats = {
        'duplicates_multi': dup_multi_count,
        'no_match': base_no_match + remaining_count,
        'male_moved': male_count if INSIGHTFACE_AVAILABLE else 0,
        'female_remaining': female_count if INSIGHTFACE_AVAILABLE else 0,
        'errors': len(debug_log),
        'moved_latin_to_numbers': moved_latin_to_numbers,
        'moved_near_duplicates': moved_near_duplicates,
    }

    print("\n" + "=" * 60)
    print("🎉 АНАЛИЗ ЗАВЕРШЁН!")
    print(f"⏱️  Время: {total_time:.1f} сек.")
    if INSIGHTFACE_AVAILABLE:
        print(f"👦 Перенесено в Парни: {stats['male_moved']}")
        print(f"👩 Осталось в Цифры: {stats['female_remaining']}")
    print(f"👥 Групп дублей: {stats['duplicates_multi']}")
    print(f"❓ Без совпадений: {stats['no_match']}")
    print(f"🔤 Латинских без дублей -> Цифры: {stats['moved_latin_to_numbers']}")
    print(f"👥 По ФИО/ФИ -> Sovpadenia: {stats['moved_near_duplicates']}")
    if debug_log:
        print("\nЛог ошибок:")
        for err in debug_log:
            print(f"  {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()