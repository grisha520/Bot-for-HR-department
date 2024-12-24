import faiss
import os
from typing import List
from sentence_transformers import SentenceTransformer

class FAISSHandler:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Инициализация обработчика FAISS.

        :param index_path: Путь к файлу FAISS индекса
        :param model_name: Название модели SentenceTransformer
        """
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = self.load_index()
        self.docs = []  # Список текстовых фрагментов для поиска

    def load_index(self) -> faiss.IndexFlatL2:
        """
        Загружает FAISS индекс или создаёт новый, если файл отсутствует.

        :return: FAISS индекс
        """
        if os.path.exists(self.index_path):
            print(f"Загрузка индекса из {self.index_path}")
            return faiss.read_index(self.index_path)
        else:
            print("Создание нового FAISS индекса.")
            return faiss.IndexFlatL2(self.dimension)

    def save_index(self):
        """Сохраняет FAISS индекс в файл."""
        faiss.write_index(self.index, self.index_path)
        print(f"Индекс сохранён в {self.index_path}")

    def add_to_index(self, embeddings: List[List[float]]):
        """
        Добавляет эмбеддинги в FAISS индекс.

        :param embeddings: Список эмбеддингов
        """
        self.index.add(embeddings)
        print(f"Добавлено {len(embeddings)} эмбеддингов в индекс.")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов.

        :param texts: Список текстов
        :return: Список эмбеддингов
        """
        return self.model.encode(texts, convert_to_numpy=True)

    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Разбивает текст на части заданного размера с перекрытием.

        :param text: Исходный текст
        :param chunk_size: Размер одной части
        :param overlap: Количество перекрывающихся символов
        :return: Список частей текста
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def add_document(self, filename: str, document_text: str):
        """
        Добавляет документ в FAISS индекс.

        :param filename: Имя файла документа
        :param document_text: Текст документа
        """
        try:
            chunks = self.split_text(document_text)
            self.docs.extend(chunks)  # Сохраняем фрагменты текста
            embeddings = self.generate_embeddings(chunks)
            self.add_to_index(embeddings)
            self.save_index()
            print(f"Документ '{filename}' успешно добавлен в индекс.")
        except Exception as e:
            print(f"Ошибка при добавлении документа: {e}")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Выполняет поиск по FAISS индексу.

        :param query: Вопрос для поиска
        :param top_k: Количество ближайших результатов
        :return: Список найденных фрагментов текста
        """
        try:
            embedding = self.generate_embeddings([query])
            distances, indices = self.index.search(embedding, top_k)
            results = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.docs):
                    results.append(self.docs[idx])
            return results
        except Exception as e:
            print(f"Ошибка при выполнении поиска: {e}")
            return []
