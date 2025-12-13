from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
import torch
from modelscope import snapshot_download
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import datetime


model_name = "Qwen/Qwen3-4B"
model_dir = snapshot_download(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.0,
    do_sample=False,
    top_p=0.9,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)


@tool(
    "get_current_date",
    description="Возвращает текущую дату, время, день недели, месяц и год."
)
def get_current_date(query: str) -> str:
    now = datetime.datetime.now()

    days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    months = [
        "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
        "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"
    ]

    return (
        f"Дата: {now.date()}\n"
        f"Время: {now.strftime('%H:%M:%S')}\n"
        f"День недели: {days[now.weekday()]}\n"
        f"Месяц: {months[now.month-1]}\n"
        f"Год: {now.year}"
    )


tools = [get_current_date]

system_prompt = """
ТЫ — АГЕНТ. ТВОЯ ФУНКЦИЯ — ОТВЕЧАТЬ НА ВОПРОСЫ О ДАТЕ И ВРЕМЕНИ.

### КАК РАБОТАТЬ:
1. Если пользователь спрашивает о дате/времени — вызывай инструмент `get_current_date`
2. ПОЛУЧИ РЕЗУЛЬТАТ от инструмента
3. ПОКАЖИ РЕЗУЛЬТАТ пользователю

### ФОРМАТ РАБОТЫ:
<tool_call>
{"tool": "get_current_date", "query": "ВОПРОС ПОЛЬЗОВАТЕЛЯ"}
</tool_call>

<tool_response>
ЗДЕСЬ БУДЕТ РЕЗУЛЬТАТ ИНСТРУМЕНТА
</tool_response>

<response>
ОТВЕТ ПОЛЬЗОВАТЕЛЮ НА ОСНОВЕ РЕЗУЛЬТАТА
</response>
"""

agent = create_react_agent(
    chat_model,
    tools,
    prompt=system_prompt
)

tests = [
    "Какая сегодня дата?",
    "Какой сейчас месяц?",
    "Какое сегодня число?",
    "Сколько сейчас времени?",
    "Сегодня какой день недели?",
    "Дата?",
    "Время?",
    "Какой сейчас год?"
]

for question in tests:
    result = agent.invoke({"messages": [
        {"role": "user", "content": question}
    ]})

    for m in result["messages"]:
        m.pretty_print()