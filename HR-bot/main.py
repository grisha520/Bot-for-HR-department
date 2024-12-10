from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import aiofiles
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from evaluate import load
import numpy as np
import torch
import sacrebleu
import time

# Инициализация FastAPI
app = FastAPI()

# Подключение HTML-шаблонов
templates = Jinja2Templates(directory="templates")

# Инициализация модели для генерации
MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Генерация конфига
generation_config = GenerationConfig(
    bos_token_id=128000,
    eos_token_id=128009,
    pad_token_id=128000,
    do_sample=True,
    max_new_tokens=256,
    repetition_penalty=1.05,
    temperature=0.1,
    top_k=5,
    top_p=0.85
)

# Инициализация SentenceTransformer
embedding_model = SentenceTransformer("deepvk/USER-bge-m3")

# Метрики для оценки
bertscore = load("bertscore")
meteor = load("meteor")
rouge = load("rouge")

# Глобальные переменные для документации и её обработанных данных
documentation_text = ""
chunks = []
chunk_embeddings = []
bm25 = None

# Функция для разбиения текста на чанки
def split_into_chunks(text, chunk_size=200):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Функция для подготовки поиска
def prepare_search_engine(text):
    global chunks, chunk_embeddings, bm25
    start_time = time.time()  # Начало замера времени
    chunks = split_into_chunks(text)
    tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    chunk_embeddings = embedding_model.encode(chunks)
    end_time = time.time()  # Конец замера времени
    print(f"Время загрузки документации: {end_time - start_time:.2f} секунд")

# Функция для гибридного поиска
def hybrid_search(user_query, bm25_weight=0.5, embedding_weight=0.5, n_results=4):
    tokenized_query = word_tokenize(user_query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    query_embedding = embedding_model.encode([user_query])
    embedding_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
    combined_scores = bm25_weight * bm25_scores + embedding_weight * embedding_scores
    top_n_indices = np.argsort(combined_scores)[::-1][:n_results]
    return [chunks[i] for i in top_n_indices]

# Функция для создания промпта
def get_prompt(user_query, context):
    return f"""Ты специалист тех поддержки, которые отвечает на вопросы пользователей по технической документации.
    Тебе нужно отвечать на вопросы строго по документации, не придумывая ничего самому.
    Отвечай на вопросы развернуто.
    Не генерируй примеры кода, которые демонстрируют, как можно обрабатывать вопросы и ответы.
    Отвечай на вопросы в деловом тоне.
    Над ответом на вопрос думаю пошагово.
    В ответах ни как не упоминай, что ты отвечаешь на вопрос по документации или какому-либо контексту.
    В ответе не упоминай вопрос пользователя.
    Не указывай никаких ссылок, если они не были заранее даны в документации.
    Если ответа нет в документации, тогда пиши я не знаю.
    content : отвечай на вопрос по этому контенту: {context}.
    user : вопрос пользователя звучит так: {user_query}"""

# Функция для генерации ответа
def rag_with_chunks(query):
    start_time = time.time()  # Начало замера времени
    retrieved_chunks = hybrid_search(query)
    prompt = get_prompt(query, retrieved_chunks)
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    end_time = time.time()  # Конец замера времени
    print(f"Время генерации ответа: {end_time - start_time:.2f} секунд")
    return response

# Вычисление метрик
def evaluate_response(prediction, reference):
    results = {}

    # BERTScore
    bertscore_results = bertscore.compute(predictions=[prediction], references=[reference], lang="ru")
    results["BERTScore"] = bertscore_results["f1"][0]

    # BLEU
    bleu_score = sacrebleu.sentence_bleu(prediction, [reference]).score
    results["BLEU"] = bleu_score

    # METEOR
    meteor_score = meteor.compute(predictions=[prediction], references=[reference])["meteor"]
    results["METEOR"] = meteor_score

    # ROUGE
    rouge_results = rouge.compute(predictions=[prediction], references=[reference])
    results["ROUGE-L"] = rouge_results["rougeL"]

    return results

# Основной роут для отображения формы
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": "", "doc_loaded": bool(chunks)})

# Роут для загрузки файла
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    global documentation_text
    async with aiofiles.open(file.filename, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    async with aiofiles.open(file.filename, "r", encoding="utf-8") as in_file:
        documentation_text = await in_file.read()
    prepare_search_engine(documentation_text)
    return templates.TemplateResponse("index.html", {"request": request, "response": "Файл успешно загружен!", "doc_loaded": True})

# Роут для обработки запроса пользователя
@app.post("/ask", response_class=HTMLResponse)
async def post_form(request: Request, question: str = Form(...)):
    global documentation_text
    reference = ""  # Это эталонный ответ для оценки
    if chunks:
        bot_response = rag_with_chunks(question)
        metrics = evaluate_response(bot_response, reference)
        print(f"Ответ: {bot_response}")
        print("Оценка метрик:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        bot_response = "Документация не загружена. Загрузите файл и повторите попытку."
    return templates.TemplateResponse("index.html", {"request": request, "response": bot_response, "doc_loaded": bool(chunks)})

import nltk
nltk.download('punkt')
