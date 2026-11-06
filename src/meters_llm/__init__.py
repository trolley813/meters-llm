import json
import os

from dotenv import load_dotenv
import gradio as gr
import ollama as ol

load_dotenv()

SYSTEM_PROMPT = (
    "Вы - ассистент контент-менеджера. Ваша задача - сформировать в формате Markdown "
    "карточку товара измерительного прибора. Она должна содержать: SEO-заголовок до 70 символов, "
    "SEO-описание на 800-1200 символов, список ключевых характеристик, 3-5 запросов для оптимизации "
    "веб-страницы карточки и блок для менеджера - кому подходит данный прибор и какие задачи решает. "
    "Входные данные подаются в виде JSON-объекта: ключ name - название, manufacturer - производитель, "
    "specs - технические характеристики в виде словаря, description - описание. Очень важно - не используйте "
    " характеристики, которых нет во входных данных, если какое-либо поле отсутствует, поставьте пометку "
    '"нет данных". Также не используйте данные, которые заведомо некорректны.'
)

client = ol.Client(host=os.environ.get("OLLAMA_HOST"))


def load_examples() -> list[list[str]]:
    examples = []
    for fn in os.listdir("examples"):
        if fn.endswith(".json"):
            with open(f"examples/{fn}") as f:
                content = json.load(f)
                examples.append(
                    [
                        content["name"],
                        content["manufacturer"],
                        json.dumps(content["specs"], ensure_ascii=False, indent=2),
                        content["description"],
                    ]
                )
    return examples


def validate(name: str, manufacturer: str, specs: dict[str, str], description: str):
    return [
        True,
        True,
        gr.validate(specs != {}, "Должна быть указана хотя бы одна характеристика"),
        gr.validate(description != "", "Описание не должно быть пустым"),
    ]


def generate(
    name: str, manufacturer: str, specs: dict[str, str], description: str
) -> tuple[str, str | None]:
    prompt = json.dumps(
        {
            "name": name,
            "manufacturer": manufacturer,
            "specs": specs,
            "description": description,
        }
    )
    try:
        response = ol.chat(
            model=os.environ.get("OLLAMA_MODEL", "qwen3.8"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "num_ctx": 16384,
                "num_predict": 16384,
            },
        )
        if "message" in response:
            content = response["message"]["content"]
            return content, content
        else:
            return f"### Произошла ошибка\n ```{response}```", None
    except Exception as e:
        return f"### Произошла ошибка\n ```{e}```", None


with gr.Blocks(title="Генератор карточек КИПиА") as demo:
    with gr.Row():
        gr.Markdown("# Создание карточки измерительного прибора с помощью LLM")
    with gr.Row():
        with gr.Column():
            input_name = gr.Textbox(label="Название прибора")
            input_manuf = gr.Textbox(label="Производитель")
            input_specs = gr.Code(label="Технические характеристики (объект JSON)")
            input_descr = gr.Textbox(label="Описание")

            submit_btn = gr.Button("Сгенерировать")

            examples = gr.Examples(
                examples=load_examples(),
                inputs=[input_name, input_manuf, input_specs, input_descr]
            )
        with gr.Column():
            with gr.Row(height="60%"):
                output_md = gr.Markdown(
                    label="Карточка товара",
                    value="""
                    # Здесь будет карточка товара
                    После генерации тут появится вся необходимая информация.
                    
                    В зависимости от модели и ИИ-сервера это может занять до нескольких минут.
                    """,
                )
            with gr.Row(height="40%"):
                output_text = gr.Textbox(
                    label="Markdown-вывод",
                    placeholder="Здесь будет текст в формате Markdown; его можно будет скопировать для дальнейшего редактирования",
                )

    submit_btn.click(
        fn=generate,
        inputs=[input_name, input_manuf, input_specs, input_descr],
        outputs=[output_md, output_text],
        validator=validate,
        api_name="generate-card",
    )


demo.launch()
