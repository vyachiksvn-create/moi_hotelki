import os
import sys
import shutil
import time
import re
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from difflib import SequenceMatcher

import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")

try:
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# === НАСТРОЙКИ ===
BASE_DIR = Path(r"C:\Foto")
CONFIG_PATH = Path(__file__).with_name("config.json")

DEFAULTS = {
    "base_directory": str(BASE_DIR),
    "folders": {
        "baza": "Baza",
        "parni": "Parni",
        "nea": "Nea",
        "sovpadenia": "Sovpadenia",
        "tsifry": "Tsifry"
    },
    "cache_file": ".embeddings_cache.pkl",
    "thresholds": {
        "gender_confidence": 0.7,
        "similarity_score": 0.78,
        "min_face_size_ratio": 0.05
    },
    "gender_mapping": {
        "male_gender_code": 0,
        "female_gender_code": 1
    },
    "performance": {
        "cpu_det_size": [320, 320],
        "gpu_det_size": [640, 640]
    }
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg.update(data)
        except Exception as e:
            print(f"⚠️ Не удалось прочитать config.json: {e}")
    return cfg


config = load_config()
BASE_DIR = Path(config["base_directory"]).resolve()
BASE_FOLDER = BASE_DIR / config["folders"]["baza"]
NUMBERS_FOLDER = BASE_DIR / config["folders"]["tsifry"]
NUMBERS_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_DUPLICATES_MULTI = BASE_DIR / "Dupes"
OUTPUT_DUPLICATES_MULTI.mkdir(parents=True, exist_ok=True)

GUYS_FOLDER = BASE_DIR / config["folders"]["parni"]
GUYS_FOLDER.mkdir(parents=True, exist_ok=True)

NEAR_DUPLICATES_FOLDER = BASE_DIR / config["folders"]["sovpadenia"]
NEAR_DUPLICATES_FOLDER.mkdir(parents=True, exist_ok=True)

NEA_FOLDER = BASE_DIR / config["folders"]["nea"]
NEA_FOLDER.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.webp', '.WEBP'}

CURRENT_INSIGHT_MODEL = 'buffalo_l'
GENDER_CONFIDENCE_THRESHOLD = float(config["thresholds"]["gender_confidence"])
SIM_THRESHOLD = float(config["thresholds"]["similarity_score"])
MIN_FACE_RATIO = float(config["thresholds"]["min_face_size_ratio"])
MALE_GENDER_CODE = int(config["gender_mapping"]["male_gender_code"])
FEMALE_GENDER_CODE = int(config["gender_mapping"]["female_gender_code"])

perf = config.get("performance", {})
CPU_DET_SIZE = tuple(perf.get("cpu_det_size", [320, 320]))
GPU_DET_SIZE = tuple(perf.get("gpu_det_size", [640, 640]))


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


def check_gender(face) -> str:
    gender = getattr(face, 'gender', None)
    gender_prob = getattr(face, 'gender_prob', None)
    if gender_prob is not None:
        confidence = float(gender_prob)
        if confidence < GENDER_CONFIDENCE_THRESHOLD:
            return 'unknown'
    if isinstance(gender, (int, np.integer)):
        g = int(gender)
        if g == MALE_GENDER_CODE:
            return 'male'
        if g == FEMALE_GENDER_CODE:
            return 'female'
    return 'unknown'


def cosine_distance(emb1, emb2) -> float:
    v1 = np.array(emb1, dtype=np.float32)
    v2 = np.array(emb2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1.0 - float(np.dot(v1, v2) / (norm1 * norm2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=np.float32).reshape(1, -1)
    b = np.array(b, dtype=np.float32).reshape(1, -1)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    val = np.dot(a, b.T)
    if isinstance(val, np.ndarray):
        val = val.item()
    return float(val / (norm_a * norm_b))


def extract_embeddings_insightface(img_bgr: np.ndarray, app: FaceAnalysis) -> Optional[np.ndarray]:
    faces = app.get(img_bgr)
    if not faces:
        return None
    faces = sort_faces_by_size(faces)
    return faces[0].embedding


def is_match(emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> Tuple[bool, float]:
    if emb1 is None or emb2 is None:
        return False, 1.0
    dist = cosine_distance(emb1, emb2)
    return dist <= SIM_THRESHOLD, dist


def has_latin(text: str) -> bool:
    has_lat = bool(re.search(r'[A-Za-z]', text))
    has_cyr = bool(re.search(r'[\u0400-\u04FF]', text))
    return has_lat and not has_cyr


def is_fio_match(fio1: str, fio2: str) -> bool:
    parts1 = normalize_fio(fio1).split()
    parts2 = normalize_fio(fio2).split()
    if len(parts1) < 2 or len(parts2) < 2:
        return False
    sim_surname = SequenceMatcher(None, parts1[0], parts2[0]).ratio()
    sim_name = SequenceMatcher(None, parts1[1], parts2[1]).ratio()
    if len(parts1) >= 3 and len(parts2) >= 3:
        sim_patr = SequenceMatcher(None, parts1[2], parts2[2]).ratio()
        return sim_surname > 0.85 and sim_name > 0.85 and sim_patr > 0.85
    return sim_surname > 0.85 and sim_name > 0.85


def normalize_name(name: str) -> str:
    text = Path(name).stem.lower()
    text = re.sub(r'[^a-zа-яё\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[\u0400-\u04FF]', text))


def normalize_fio(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r'\s*\(\d+\)\s*', '', name)
    name = re.sub(r'[_\-]\d+$', '', name)
    name = re.sub(r'[^\w\s\-]', '', name)
    return name.strip().lower()


def has_fio(filename: str) -> bool:
    name = Path(filename).stem
    words = re.findall(r'[А-Яа-яЁё\-]+', name)
    valid_words = [w for w in words if len(w) > 3]
    return len(valid_words) >= 2


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
    if not src.exists():
        return None
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


def find_visual_duplicates(files_data: List[dict], threshold: float = 0.78) -> List[List[dict]]:
    groups: List[List[dict]] = []
    used = set()
    for i, file1 in enumerate(files_data):
        if i in used:
            continue
        if file1.get('embedding') is None:
            continue
        current_group = [file1]
        used.add(i)
        for j, file2 in enumerate(files_data):
            if i == j or j in used:
                continue
            if file2.get('embedding') is None:
                continue
            g1 = file1.get('gender', 'unknown')
            g2 = file2.get('gender', 'unknown')
            if g1 != 'unknown' and g2 != 'unknown' and g1 != g2:
                continue
            sim = cosine_similarity(file1['embedding'], file2['embedding'])
            if sim >= threshold:
                current_group.append(file2)
                used.add(j)
        if len(current_group) > 1:
            groups.append(current_group)
    return groups


def finalize_pipeline(base_dir: Path):
    sovpadeniya_dir = base_dir / "Sovpadenia"
    baza_after_dir = base_dir / "Baza после"
    baza_after_dir.mkdir(parents=True, exist_ok=True)
    print("\n[final] Финальная очистка...")
    moved = 0
    if not sovpadeniya_dir.exists():
        return moved
    for item in sovpadeniya_dir.iterdir():
        try:
            if item.is_file():
                dst = baza_after_dir / item.name
                if not dst.exists():
                    shutil.move(str(item), str(dst))
                    moved += 1
            elif item.is_dir():
                files = [f for f in item.iterdir() if f.is_file()]
                if len(files) == 1:
                    dst = baza_after_dir / files[0].name
                    if not dst.exists():
                        shutil.move(str(files[0]), str(dst))
                        moved += 1
                    try:
                        item.rmdir()
                    except Exception:
                        pass
        except Exception as e:
            print(f"  ❌ Ошибка финализации {item}: {e}")
    print(f"  📦 Файлов возвращено в Baza после: {moved}")
    return moved


def main():
    try:
        os.environ['PYTHONUTF8'] = '1'
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if not INSIGHTFACE_AVAILABLE:
        print("❌ InsightFace не найден. Установите: pip install insightface")
        sys.exit(1)

    start_time = time.time()
    BAZA_AFTER = BASE_DIR / "Baza после"
    BAZA_AFTER.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 ОРГАНИЗАЦИЯ ФОТО: БАЗА → ПАРНИ / NEA / СОВПАДЕНИЯ / ЦИФРЫ / BAZA POSLE")
    print(f"📁 База: {BASE_FOLDER}")
    print(f"📄 Файл: {Path(__file__).resolve()}")
    print(f"👦 Парни: {GUYS_FOLDER}")
    print(f"👩 Nea: {NEA_FOLDER}")
    print(f"👥 Совпадения: {NEAR_DUPLICATES_FOLDER}")
    print(f"🔢 Цифры: {NUMBERS_FOLDER}")
    print(f"📁 Baza после: {BAZA_AFTER}")
    print("=" * 60)

    for f in OUTPUT_DUPLICATES_MULTI.iterdir():
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass

    debug_log = []
    processed_files = set()

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
            det_size = GPU_DET_SIZE if 'CUDAExecutionProvider' in providers else CPU_DET_SIZE
            app = FaceAnalysis(name=CURRENT_INSIGHT_MODEL, root=os.path.join(os.path.expanduser('~'), '.insightface', 'models'), providers=providers)
            app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1, det_size=det_size)
            print(f"✅ InsightFace initialized ({CURRENT_INSIGHT_MODEL})")
        except Exception as e:
            print(f"❌ Ошибка инициализации InsightFace: {e}")
            sys.exit(1)

    # ============================================================
    # ШАГ 1. Сначала выносим ФИО в Совпадения по нечеткому совпадению
    # ============================================================
    print("\n[1/5] ФИО/ФИ группировка (fuzzy)...")
    base_images = get_image_paths(BASE_FOLDER)
    print(f"  Найдено файлов в базе: {len(base_images)}")

    fio_groups: Dict[str, List[Path]] = {}
    fio_single = []
    for img in base_images:
        if not img.exists():
            continue
        if not has_fio(img.name):
            continue
        norm = normalize_fio(img.name)
        matched_key = None
        for key in fio_groups.keys():
            if is_fio_match(norm, key):
                matched_key = key
                break
        if matched_key:
            fio_groups[matched_key].append(img)
        else:
            fio_groups[norm] = [img]

    skip_gender = set()
    for key, members in fio_groups.items():
        if len(members) < 2:
            fio_single.extend(members)
            continue
        members = sorted(members, key=lambda p: p.name)
        target_dir = NEAR_DUPLICATES_FOLDER / members[0].stem
        target_dir.mkdir(parents=True, exist_ok=True)
        for member in members:
            if safe_move(member, target_dir, member.name):
                skip_gender.add(member.name)
    print(f"  👥 Групп по ФИО/ФИ: {len(fio_groups)}")
    print(f"  📄 Одиночных ФИО: {len(fio_single)}")

    # ============================================================
    # ШАГ 2. Гендерныйsplit по оставшимся в Baza
    # ============================================================
    print("\n[2/5] Разделение по полу (База → Парни / Nea)...")
    remaining = [f for f in get_image_paths(BASE_FOLDER) if f.exists() and f.name not in skip_gender]
    male_count = 0
    female_count = 0
    no_gender_count = 0
    female_images = []
    gender_map: Dict[str, str] = {}

    for idx, img in enumerate(remaining, 1):
        if idx % 10 == 0 or idx == len(remaining):
            print(f"  Обработка: {idx}/{len(remaining)}")
        img_bgr = load_image(img)
        if img_bgr is None:
            no_gender_count += 1
            gender_map[img.name] = 'unknown'
            continue
        try:
            faces = app.get(img_bgr) if app is not None else []
            if not faces:
                no_gender_count += 1
                gender_map[img.name] = 'unknown'
                continue
            faces = sort_faces_by_size(faces)
            gender = check_gender(faces[0])
            if gender == 'unknown':
                no_gender_count += 1
                gender_map[img.name] = 'unknown'
                continue
            gender_map[img.name] = gender
            if gender == 'male':
                if safe_move(img, GUYS_FOLDER, img.name):
                    male_count += 1
            else:
                female_images.append(img)
        except Exception as e:
            debug_log.append(f"❌ Ошибка обработки {img.name}: {e}")
            no_gender_count += 1
            gender_map[img.name] = 'unknown'

    print(f"  👦 Мужчин перенесено в Парни: {male_count}")
    print(f"  👩 Женщин на проверке дублей: {len(female_images)}")
    print(f"  ❓ Пол не определён: {no_gender_count}")

    # ============================================================
    # ШАГ 3. Визуальная дедупликация только женских лиц
    # ============================================================
    print("\n[3/5] Визуальная дедупликация...")
    visual_data = []
    for img in female_images:
        if not img.exists():
            continue
        img_bgr = load_image(img)
        if img_bgr is None:
            continue
        try:
            faces = app.get(img_bgr) if app is not None else []
            if not faces:
                continue
            faces = sort_faces_by_size(faces)
            emb = faces[0].embedding
            if emb is None:
                continue
            visual_data.append({
                'filename': img.name,
                'path': img,
                'embedding': emb,
                'gender': gender_map.get(img.name, 'female')
            })
        except Exception as e:
            debug_log.append(f"❌ Ошибка duplicate search {img.name}: {e}")

    dup_groups = find_visual_duplicates(visual_data, threshold=SIM_THRESHOLD)
    dup_multi_count = 0
    moved_duplicates = 0
    for group in dup_groups:
        if len(group) < 2:
            continue
        dup_multi_count += 1
        group_sorted = sorted(group, key=lambda x: x['filename'])
        representative = Path(group_sorted[0]['filename']).stem
        target_dir = NEAR_DUPLICATES_FOLDER / representative
        target_dir.mkdir(parents=True, exist_ok=True)
        for member in group_sorted[1:]:
            if member['path'].exists() and member['filename'] not in processed_files:
                if safe_move(member['path'], target_dir, member['filename']):
                    moved_duplicates += 1
                    processed_files.add(member['filename'])

    print(f"  👥 Групп дублей: {dup_multi_count}")
    print(f"  📦 Файлов перемещено в Совпадения: {moved_duplicates}")

    # ============================================================
    # ШАГ 4. Латинские имена без точных совпадений → Цифры
    # ============================================================
    print("\n[4/5] Латинские имена → Цифры...")
    moved_latin = 0
    remaining_latin = []
    for folder in [BASE_FOLDER]:
        if folder.exists():
            remaining_latin.extend(get_image_paths(folder))

    for img in remaining_latin:
        if not img.exists() or img.name in skip_gender:
            continue
        if has_latin(img.name):
            if safe_move(img, NUMBERS_FOLDER, img.name):
                moved_latin += 1
    print(f"  🔤 Латинских файлов перенесено в Цифры: {moved_latin}")

    # ============================================================
    # ШАГ 5. Финальная сборка в Baza после и очистка
    # ============================================================
    print("\n[5/5] Финальная сборка...")
    final_count = 0
    for img in get_image_paths(BASE_FOLDER):
        if not img.exists() or img.name in skip_gender:
            continue
        dst = BAZA_AFTER / img.name
        if not dst.exists():
            try:
                shutil.move(str(img), str(dst))
                final_count += 1
            except Exception as e:
                debug_log.append(f"❌ Ошибка финализации {img.name}: {e}")
    print(f"  📦 Файлов перенесено в Baza после: {final_count}")

    for member in fio_single:
        if member.exists():
            dst = BAZA_AFTER / member.name
            if not dst.exists():
                try:
                    shutil.move(str(member), str(dst))
                except Exception as e:
                    debug_log.append(f"❌ Ошибка финализации ФИО {member.name}: {e}")

    # Очистка одиночных файлов/папок в Совпадениях
    finalized = finalize_pipeline(BASE_DIR)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 ОРГАНИЗАЦИЯ ЗАВЕРШЕНА!")
    print(f"⏱️  Время: {total_time:.1f} сек.")
    print(f"👦 Перенесено в Парни: {male_count}")
    print(f"👩 Женщин обработано: {len(female_images) - moved_duplicates}")
    print(f"👥 Групп дублей: {dup_multi_count}")
    print(f"🔤 Латинских → Цифры: {moved_latin}")
    print(f"📁 Файлов в Baza после: {final_count}")
    if debug_log:
        print("\nЛог ошибок:")
        for err in debug_log:
            print(f"  {err}")
    print("=" * 60)


if __name__ == "__main__":
    main()
