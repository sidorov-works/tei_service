# shared/utils/download_models.py

import os
from sentence_transformers import SentenceTransformer
from shared.config import config
from pathlib import Path

# Пути для сохранения
os.makedirs(config.MODELS_PATH, exist_ok=True)

def download_sentence_transformers():
    """Скачивает и сохраняет модель для эмбеддингов RAG."""
    model_name = config.EMBEDDING_MODEL['model']
    save_path = Path(config.MODELS_PATH) / config.EMBEDDING_MODEL['subdir'] / model_name
    
    if not save_path.exists():
        print(f"Скачивание {model_name}...")
        model = SentenceTransformer(model_name)
        model.save(str(save_path))
        print(f"Модель сохранена в {save_path}")
    else:
        print(f"Модель {model_name} уже загружена.")


if __name__ == "__main__":
    download_sentence_transformers()
    print("Все модели готовы!")