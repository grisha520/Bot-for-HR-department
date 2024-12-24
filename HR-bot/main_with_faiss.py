from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import aiofiles
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from evaluate import load
import sacrebleu
import time
import torch
from faiss_handler import FAISSHandler

INDEX_PATH = "faiss_index"
faiss_handler = FAISSHandler(index_path=INDEX_PATH)

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

# Метрики для оценки
bertscore = load("bertscore")
meteor = load("meteor")
rouge = load("rouge")

# Глобальный обработчик FAISS
faiss_handler = FAISSHandler(INDEX_PATH)

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
    retrieved_chunks = faiss_handler.search(query)
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
    return templates.TemplateResponse("index.html", {"request": request, "response": "", "doc_loaded": bool(faiss_handler.docs)})

# Роут для загрузки файла
@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    async with aiofiles.open(file.filename, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    async with aiofiles.open(file.filename, "r", encoding="utf-8") as in_file:
        documentation_text = await in_file.read()
    faiss_handler.add_document(file.filename, documentation_text)
    return "Файл успешно загружен!"

# Роут для обработки запроса пользователя
@app.post("/ask", response_class=HTMLResponse)
async def post_form(request: Request, question: str = Form(...)):
    reference = "«Информационные ресурсы» 1. Почта 2. Диск U и диск T 3. Sharepoint 4. Стафф 5. Корпоративный TГ- канал 6. Бизнес-ТГ-канал 7. ВК 8. Instagram 9. Youtube"  # Это эталонный ответ для оценки
    if faiss_handler.docs:
        bot_response = rag_with_chunks(question)
        metrics = evaluate_response(bot_response, reference)
        print(f"Ответ: {bot_response}")
        print("Оценка метрик:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        bot_response = "Документация не загружена. Загрузите файл и повторите попытку."
    return bot_response