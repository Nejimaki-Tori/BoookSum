from typing import List
from evaluate import load as hf_load
from sentence_transformers import SentenceTransformer
from ignite.metrics import RougeL
from utils import lemmatize_text, LlmCompleter, AsyncList, oclient, model
import re
import torch


rouge = hf_load('rouge')
bert_score = hf_load('bertscore', model_type='deepvk/USER-bge-m3')
encoder = SentenceTransformer('deepvk/USER-bge-m3')


def rouge_L(pred: str, ref: str) -> float:
    m = RougeL(multiref="average")

    lem_candidate = lemmatize_text(pred).split()
    lem_references = lemmatize_text(ref).split()

    m.update(([lem_candidate], [lem_references]))

    return round(m.compute()['Rouge-L-F'], 4)


def bert_f1(pred: str, ref: str) -> float:
    return round(bert_score.compute(predictions=[pred], references=[ref], lang='ru')['f1'][0], 3)


def similarity(a: str, b: str) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_1 = encoder.encode(a, device=device)
    emb_2 = encoder.encode(b, device=device)

    return round(float(encoder.similarity(emb_1, emb_2).item()), 3)


async def compute_coverage(
        questions: List[str],
        summary: str,
        client: LlmCompleter,
        positive_choice: str = "Д",
        negative_choice: str = "Н"
) -> tuple[float, list[int]]:
    key_q_prompt = """Вопрос:
    {}
    Текст:
    {}
    Содержится ли в этом тексте ответ на вопрос?
    Начни ответ с {} или {}
    """

    probs = AsyncList()

    for q in questions:
        prompt = key_q_prompt.format(q, summary, positive_choice, negative_choice)
        probs.append(client.get_probability(prompt, rep_penalty=1.0, max_tokens=10))

    await probs.complete_couroutines(batch_size=20)
    results = await probs.to_list()

    # подсчитываем покрытие
    flags: List[int] = []

    print(results)

    for res in results:
        if negative_choice in res:
            prob = 1 - res[negative_choice]
        elif positive_choice in res:
            prob = res[positive_choice]
        else:
            prob = 0.0

        print(prob)
        flags.append(1 if prob >= 0.75 else 0)

    coverage = sum(flags) / len(flags) if flags else 0.0

    return round(coverage, 3), flags


def generate_key_questions(ref_annotation, model="llama3-70b"):
    prompt = f"""На основе данной аннотации сформируй несколько ключевых вопросов, ответы на которые можно однозначно дать, зная содержание аннотации:
            {ref_annotation}
            ---
            Ключевые вопросы нужно писать по порядку, начиная каждый вопрос с новой строки. Кроме ключевых вопросов ничего писать не нужно.
            """

    res = oclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=4000,
        extra_body={
            "repetition_penalty": 1.0,
            "guided_choice": None,
            "add_generation_prompt": True,
            "guided_regex": None
        }
    )

    raw = res.choices[0].message.content.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    questions = [re.sub(r'^\s*(?:\d+[\.\)]|[-•])\s*', '', line) for line in lines]

    return questions


def generate_key_answers(ref_annotation, key_q, model="llama3-70b"):
    prompt = f"""На основе данной аннотации сформируй ответы на несколько ключевых вопросов:
            {ref_annotation}

            Ключевые вопросы:
            {key_q}
            ---
            Ответы на ключевые вопросы нужно писать по порядку, начиная каждый ответ с новой строки. Не пиши вопросы, пиши только ответы на ключевые вопросы.
            """

    res = oclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=4000,
        extra_body={
            "repetition_penalty": 1.0,
            "guided_choice": None,
            "add_generation_prompt": True,
            "guided_regex": None
        }
    )

    raw = res.choices[0].message.content.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    answers = [re.sub(r'^\s*(?:\d+[\.\)]|[-•])\s*', '', line) for line in lines]

    return answers


def compute_answer_similarity(questions, summary, cov_flags, reference_answers):
    prompt = f"""На основе данной аннотации сформируй ответы на несколько ключевых вопросов:
            {summary}

            Ключевые вопросы:
            {questions}
            ---
            Ответы на ключевые вопросы нужно писать по порядку, начиная каждый ответ с новой строки. Не пиши вопросы, пиши только ответы на ключевые вопросы.
            """

    res = oclient.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        max_tokens=4000,
        extra_body={
            "repetition_penalty": 1.0,
            "guided_choice": None,
            "add_generation_prompt": True,
            "guided_regex": None
        }
    )

    raw = res.choices[0].message.content.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    gen_answers = [re.sub(r'^\s*(?:\d+[\.\)]|[-•])\s*', '', line) for line in lines]

    print(gen_answers)

    sims: List[float] = []

    for q, flag, gen, gold in zip(questions, cov_flags, gen_answers, reference_answers):
        if flag == 0:
            sims.append(0.0)
        else:
            sims.append(similarity(gen, gold))

    print(sims)

    return round(sum(sims) / len(sims) if sims else 0.0, 3)