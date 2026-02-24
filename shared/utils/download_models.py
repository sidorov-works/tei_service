# shared/utils/download_models.py

from sentence_transformers import SentenceTransformer
from shared.config import config

def download_sentence_transformers():
    """Скачивает и сохраняет модель для эмбеддингов."""
    model_info = config.EMBEDDING_MODEL
    model_name = model_info.name
    save_path = config.MODEL_PATH / model_name
    
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