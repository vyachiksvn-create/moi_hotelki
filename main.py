import os
import sys
import json
import pickle
import logging
import shutil
import argparse
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

class Config:
    def __init__(self, path="config.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.base_dir = Path(self.data["base_directory"]).resolve()
        self.folders = {k: self.base_dir / v for k, v in self.data["folders"].items()}
        self.cache_file = self.base_dir / self.data["cache_file"]
        
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)

class FaceAnalyzer:
    def __init__(self):
        logging.info("Инициализация модели InsightFace (buffalo_l)...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
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

    @staticmethod
    def _parse_gender(face) -> Tuple[Optional[int], float, float]:
        if not hasattr(face, 'gender'):
            return None, 0.0, 0.0
        gender = face.gender
        gender_prob = getattr(face, 'gender_prob', None)
        
        if isinstance(gender, (int, np.integer)):
            gender = int(gender)
        else:
            return None, 0.0, 0.0
        
        if isinstance(gender_prob, (list, np.ndarray)) and len(gender_prob) >= 2:
            female_prob = float(gender_prob[0])
            male_prob = float(gender_prob[1])
            confidence = max(female_prob, male_prob)
            male_confidence = male_prob if gender == 1 else female_prob
            return gender, confidence, male_confidence
        elif isinstance(gender_prob, (int, float, np.floating)):
            prob = float(gender_prob)
            male_confidence = prob if gender == 1 else (1.0 - prob)
            return gender, prob, male_confidence
        else:
            return gender, 0.5, 0.5 if gender == 1 else 0.0

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
                    "embedding": f["embedding"].tolist()
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
    threshold = cfg.data["thresholds"]["gender_confidence"]
    
    files = [f for f in src_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    moved_count = 0
    for img_path in tqdm(files, desc="Анализ пола"):
        faces = analyzer.get_faces(img_path)
        male_faces = [
            f for f in faces 
            if f["gender"] == 1 and f["male_confidence"] >= threshold
        ]
        
        if male_faces:
            best_male = max(male_faces, key=lambda x: x["male_confidence"])
            logging.debug(f"Мужское лицо в {img_path.name}: уверенность={best_male['male_confidence']:.2f}")
            file_mgr.safe_move(img_path, dst_dir)
            moved_count += 1
    
    analyzer._save_cache()
    logging.info(f"Отбор мужских лиц завершен! Перемещено файлов: {moved_count}")

def find_duplicates(source_folder_key):
    src_dir = cfg.folders[source_folder_key]
    dst_dir = cfg.folders["sovpadenia"]
    sim_threshold = cfg.data["thresholds"]["similarity_score"]
    
    files = [f for f in src_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    logging.info(f"Загрузка эмбеддингов из {src_dir.name} ({len(files)} файлов)...")
    
    embeddings_data = []
    for img_path in tqdm(files, desc="Извлечение признаков"):
        faces = analyzer.get_faces(img_path)
        for i, face in enumerate(faces):
            embeddings_data.append({
                "path": img_path,
                "face_idx": i,
                "embedding": np.array(face["embedding"])
            })

    logging.info("Попарное сравнение...")
    processed_pairs = set()
    moved_files = set()
    
    for i in tqdm(range(len(embeddings_data)), desc="Поиск совпадений"):
        for j in range(i + 1, len(embeddings_data)):
            item1, item2 = embeddings_data[i], embeddings_data[j]
            
            pair_key = tuple(sorted([str(item1["path"]), str(item2["path"])]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            sim = cosine_similarity([item1["embedding"]], [item2["embedding"]])[0][0]
            
            if sim >= sim_threshold:
                logging.info(f"Совпадение: {item1['path'].name} и {item2['path'].name} ({sim:.2f})")
                
                if str(item1["path"]) not in moved_files:
                    file_mgr.safe_move(item1["path"], dst_dir, suffix="_match1")
                    moved_files.add(str(item1["path"]))
                
                if str(item2["path"]) not in moved_files:
                    file_mgr.safe_move(item2["path"], dst_dir, suffix="_match2")
                    moved_files.add(str(item2["path"]))
                
                file_mgr.add_report(item1["path"], item2["path"], sim)

    analyzer._save_cache()
    file_mgr.save_report()
    logging.info("Поиск дубликатов завершен!")

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
    args = parser.parse_args()
    
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
