import os
import sys
import json
import pickle
import logging
import shutil
import argparse
import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd

try:
    os.environ['PYTHONUTF8'] = '1'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log", encoding="utf-8"), logging.StreamHandler()]
)

def has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[\u0400-\u04FF]', text))

class Config:
    def __init__(self, path="config.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.base_dir = Path(self.data["base_directory"]).resolve()
        self.folders = {k: self.base_dir / v for k, v in self.data["folders"].items()}
        self.cache_file = self.base_dir / self.data["cache_file"]
        self.male_gender_code = int(self.data.get("gender_mapping", {}).get("male_gender_code", 0))
        self.female_gender_code = int(self.data.get("gender_mapping", {}).get("female_gender_code", 1))
        
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        if "nea" not in self.folders:
            self.folders["nea"] = self.base_dir / "Nea"
            self.folders["nea"].mkdir(parents=True, exist_ok=True)

class FaceAnalyzer:
    def __init__(self):
        logging.info("Инициализация модели InsightFace (buffalo_l)...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)

        perf = cfg.data.get("performance", {})
        cpu_det_size = tuple(perf.get("cpu_det_size", [320, 320]))
        gpu_det_size = tuple(perf.get("gpu_det_size", [640, 640]))

        try:
            import onnxruntime as ort
            has_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        except ImportError:
            has_gpu = False

        det_size = gpu_det_size if has_gpu else cpu_det_size
        logging.info(f"det_size={det_size}, providers={providers if has_gpu else ['CPUExecutionProvider']}")

        self.app.prepare(ctx_id=0 if has_gpu else -1, det_size=det_size)
        self.cache = self._load_cache()

    def _load_cache(self):
        if Path(cfg.cache_file).exists():
            try:
                with open(cfg.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"Ошибка загрузки кэша: {e}. Создаем новый.")
        return {}

    def _save_cache(self):
        with open(cfg.cache_file, "wb") as f:
            pickle.dump(self.cache, f)

    def _get_file_hash(self, filepath):
        mtime = os.path.getmtime(filepath)
        size = os.path.getsize(filepath)
        return f"{size}_{mtime}"

    def _parse_gender(self, face) -> Tuple[Optional[int], float, float]:
        if not hasattr(face, 'gender'):
            return None, 0.0, 0.0
        gender = face.gender
        gender_prob = getattr(face, 'gender_prob', None)
        
        if isinstance(gender, (int, np.integer)):
            gender = int(gender)
        else:
            return None, 0.0, 0.0
        
        male_code = cfg.male_gender_code
        female_code = cfg.female_gender_code
        
        if isinstance(gender_prob, (list, np.ndarray)) and len(gender_prob) >= 2:
            female_prob = float(gender_prob[0])
            male_prob = float(gender_prob[1])
            confidence = max(female_prob, male_prob)
            male_confidence = male_prob
            return gender, confidence, male_confidence
        elif isinstance(gender_prob, (int, float, np.floating)):
            prob = float(gender_prob)
            male_confidence = prob if gender == male_code else (1.0 - prob)
            return gender, prob, male_confidence
        else:
            if gender == male_code:
                return gender, 1.0, 1.0
            else:
                return gender, 1.0, 0.0

    def get_faces(self, img_path):
        file_hash = self._get_file_hash(img_path)
        if str(img_path) in self.cache and self.cache[str(img_path)]["hash"] == file_hash:
            return self.cache[str(img_path)]["faces"]

        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            logging.warning(f"Не удалось прочитать: {img_path}")
            return []

        faces = self.app.get(img)
        result = []
        img_height, img_width = img.shape[:2]

        for face in faces:
            bbox = face.bbox
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            img_area = img_height * img_width
            
            if (face_area / img_area) < cfg.data["thresholds"]["min_face_size_ratio"]:
                continue

            gender, gender_conf, male_confidence = self._parse_gender(face)

            result.append({
                "gender": gender,
                "gender_conf": gender_conf,
                "male_confidence": male_confidence,
                "embedding": face.embedding,
                "bbox": bbox
            })

        self.cache[str(img_path)] = {
            "hash": file_hash,
            "faces": [
                {
                    "gender": f["gender"],
                    "gender_conf": float(f["gender_conf"]),
                    "male_confidence": float(f["male_confidence"]),
                    "embedding": f["embedding"].tolist(),
                    "bbox": f["bbox"].tolist() if f.get("bbox") is not None else None
                } for f in result
            ]
        }
        return result

class FileManager:
    def __init__(self):
        self.report_data = []

    def safe_move(self, src, dst_dir, suffix=""):
        src = Path(src)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        stem = src.stem + suffix
        ext = src.suffix
        dst = dst_dir / f"{stem}{ext}"
        
        counter = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_{counter}{ext}"
            counter += 1
            
        try:
            shutil.move(str(src), str(dst))
            return dst
        except Exception as e:
            logging.error(f"Ошибка при перемещении {src}: {e}")
            return None

    def add_report(self, file1, file2, similarity):
        self.report_data.append({
            "File_1": str(file1),
            "File_2": str(file2),
            "Similarity": f"{similarity:.4f}"
        })

    def save_report(self):
        if cfg.data["generate_report"] and self.report_data:
            df = pd.DataFrame(self.report_data)
            report_path = cfg.base_dir / "duplicates_report.csv"
            df.to_csv(report_path, index=False, encoding="utf-8-sig")
            logging.info(f"Отчет сохранен: {report_path}")

cfg = Config()
analyzer = FaceAnalyzer()
file_mgr = FileManager()

def test_gender_detection():
    logging.info("=== ТЕСТ ОПРЕДЕЛЕНИЯ ПОЛА ===")
    src_dir = cfg.folders["baza"]
    files = list(src_dir.glob("*.jpg"))[:5]
    if not files:
        files = list(src_dir.glob("*.jpeg"))[:5]
    if not files:
        files = list(src_dir.glob("*.png"))[:5]
    
    for img_path in files:
        faces = analyzer.get_faces(img_path)
        print(f"\n{img_path.name}:")
        for i, face in enumerate(faces):
            print(f"  Лицо {i+1}:")
            print(f"    gender (raw): {face['gender']}")
            print(f"    gender_conf: {face['gender_conf']:.3f}")
            print(f"    male_confidence: {face['male_confidence']:.3f}")
            print(f"    bbox: {face['bbox']}")

    analyzer._save_cache()
    logging.info("Тест завершен. Кэш обновлен.")

def separate_guys():
    logging.info("Запуск отбора мужских лиц...")
    src_dir = cfg.folders["baza"]
    dst_dir = cfg.folders["parni"]
    no_face_dir = cfg.folders["nea"]
    threshold = cfg.data["thresholds"]["gender_confidence"]
    
    files = [f for f in src_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    moved_count = 0
    no_face_count = 0
    female_count = 0
    low_conf_count = 0
    
    for img_path in tqdm(files, desc="Анализ пола"):
        faces = analyzer.get_faces(img_path)
        
        if not faces:
            file_mgr.safe_move(img_path, no_face_dir)
            no_face_count += 1
            logging.debug(f"Baza->Nea (no face): {img_path.name}")
            continue
        
        best = max(faces, key=lambda x: x["male_confidence"])
        gender = best["gender"]
        male_conf = best["male_confidence"]
        gender_conf = best["gender_conf"]
        
        if gender == cfg.male_gender_code and male_conf >= threshold:
            file_mgr.safe_move(img_path, dst_dir)
            moved_count += 1
            logging.debug(f"Baza->Parni: {img_path.name} gender={gender} male_conf={male_conf:.3f}")
        elif gender == cfg.female_gender_code:
            female_count += 1
            logging.debug(f"Baza->stay (female): {img_path.name} gender={gender} male_conf={male_conf:.3f}")
        else:
            low_conf_count += 1
            logging.debug(f"Baza->stay (low conf): {img_path.name} gender={gender} male_conf={male_conf:.3f}")
    
    analyzer._save_cache()
    logging.info(f"Отбор мужских лиц завершен! Parni: {moved_count}, Nea: {no_face_count}, Female skipped: {female_count}, Low conf skipped: {low_conf_count}")

def find_duplicates(source_folder_key):
    src_dir = cfg.folders[source_folder_key]
    dst_dir = cfg.folders["sovpadenia"]
    no_face_dir = cfg.folders["nea"]
    sim_threshold = cfg.data["thresholds"]["similarity_score"]
    
    files = [f for f in src_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    logging.info(f"Загрузка эмбеддингов из {src_dir.name} ({len(files)} файлов)...")
    
    embeddings_data = []
    moved_no_face = 0
    for img_path in tqdm(files, desc="Извлечение признаков"):
        faces = analyzer.get_faces(img_path)
        if not faces:
            file_mgr.safe_move(img_path, no_face_dir)
            moved_no_face += 1
            continue
        for i, face in enumerate(faces):
            embeddings_data.append({
                "path": img_path,
                "face_idx": i,
                "embedding": np.array(face["embedding"])
            })

    logging.info("Попарное сравнение...")
    processed_pairs = set()
    moved_files = set()
    groups_found = 0
    
    for i in tqdm(range(len(embeddings_data)), desc="Поиск совпадений"):
        if embeddings_data[i]["path"] in moved_files:
            continue
        group = [embeddings_data[i]]
        for j in range(i + 1, len(embeddings_data)):
            if embeddings_data[j]["path"] in moved_files:
                continue
            item1, item2 = embeddings_data[i], embeddings_data[j]
            pair_key = tuple(sorted([str(item1["path"]), str(item2["path"])]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            sim = cosine_similarity([item1["embedding"]], [item2["embedding"]])[0][0]
            if sim >= sim_threshold:
                group.append(embeddings_data[j])

        if len(group) >= 2:
            representative = group[0]["path"].stem
            has_cyrillic_name = any(has_cyrillic(member["path"].name) for member in group)
            
            if has_cyrillic_name:
                target_dir = cfg.folders["baza"] / representative
                logging.info(f"Кириллица в названии: {len(group)} файлов -> Baza/{target_dir.name}")
            else:
                target_dir = dst_dir / representative
                
            target_dir.mkdir(parents=True, exist_ok=True)
            moved = 0
            for member in group:
                if file_mgr.safe_move(member["path"], target_dir):
                    moved += 1
                    moved_files.add(member["path"])
            groups_found += 1
            logging.info(f"Совпадение ({sim:.2f}): {len(group)} файлов -> {target_dir.name}")
            for member in group[1:]:
                file_mgr.add_report(group[0]["path"], member["path"], sim)

    # Move non-duplicate files to Baza if source is not Baza
    if source_folder_key != "baza":
        moved_to_baza = 0
        for item in embeddings_data:
            if item["path"] not in moved_files and item["path"].exists():
                if file_mgr.safe_move(item["path"], cfg.folders["baza"]):
                    moved_to_baza += 1
        if moved_to_baza:
            logging.info(f"Файлов без дубликатов перемещено в Baza: {moved_to_baza}")

    analyzer._save_cache()
    file_mgr.save_report()
    logging.info(f"Поиск дубликатов завершен! Групп: {groups_found}, без лиц в Nea: {moved_no_face}")

def auto_pipeline():
    logging.info("=== ЗАПУСК ПОЛНОГО АВТОМАТИЧЕСКОГО ПАЙПЛАЙНА ===")
    separate_guys()
    logging.info("--- Переход к поиску дубликатов среди отобранных ---")
    find_duplicates("parni")
    logging.info("=== ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО ===")

def main():
    parser = argparse.ArgumentParser(description="MOI HOTELKI - автоматическая обработка фото")
    parser.add_argument(
        "--mode", 
        type=int, 
        choices=[1, 2, 3, 4], 
        default=4,
        help="Режим работы: 1=отбор мужских лиц, 2=дубликаты в Parni, 3=дубликаты в Baza, 4=полный пайплайн (по умолчанию)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тест определения пола на первых 5 фото из Baza"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Очистить папки Parni, Sovpadenia, Nea перед запуском"
    )
    args = parser.parse_args()
    
    if args.reset:
        for key in ["parni", "sovpadenia", "nea"]:
            folder = cfg.folders.get(key)
            if folder and folder.exists():
                for f in folder.rglob("*"):
                    if f.is_file():
                        try:
                            src = cfg.folders.get("baza")
                            if src and src.exists():
                                shutil.move(str(f), str(src / f.name))
                            else:
                                f.unlink()
                        except Exception as e:
                            logging.warning(f"Не удалось сбросить {f}: {e}")
        logging.info("Папки сброшены. Файлы возвращены в Baza.")
    
    print("="*60)
    print(" MOI HOTELKI v2.3 - AVTOMATICHESKIY REZHIM")
    print("="*60)
    print(f"Базовая папка: {cfg.base_dir}")
    print(f"Запущен режим: {args.mode}")
    print("="*60)
    
    if args.test:
        test_gender_detection()
        return
    
    if args.mode == 1:
        separate_guys()
    elif args.mode == 2:
        find_duplicates("parni")
    elif args.mode == 3:
        find_duplicates("baza")
    elif args.mode == 4:
        auto_pipeline()
    
    print("\nГотово! Программа завершила работу.")

if __name__ == "__main__":
    if not cfg.base_dir.exists():
        cfg.base_dir.mkdir(parents=True)
        baza_dir = cfg.folders["baza"]
        baza_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Создана базовая папка: {cfg.base_dir}")
        logging.info(f"Пожалуйста, поместите ваши фото в папку '{baza_dir}' и запустите скрипт снова.")
    else:
        main()
