import re
from utils import AsyncList, extract_response
import random
import math
from typing import List, Dict
from metrics import encoder
import numpy as np
import faiss

BLUEPRINT_PROMPT = """Для следующего отрывка текста создайте план, обязательно состоящий из последовательности вопросов и ответов (не более 15 пар, лучше использовать только ключевые вопросы), которые помогут выделить основные события, персонажей и ключевые моменты. Создавайте только план, не добавляя ничего лишнего. Убедитесь, что каждый вопрос обязательно сопровождается четким и кратким ответом.

Текст:
---
{}
---
"""

QUESTIONS_PROMPT = """Сформулируй один обобщённый ключевой вопрос для группы вопросов.

Вопросы:
{questions_str} 

Не пиши ничего, кроме обобщённого ключевого вопроса.
"""

SUMMARIZE_BLUEPRINT_PROMPT = """
Используя следующий план из вопросов и ответов, создайте краткое содержание представленного далее текста. Текст должен быть последовательным, структурированным. Убедись, что текст логически связан и сохраняет важные элементы исходного контекста. Не добавляй ничего лишнего в ответе, только текст краткого содержания.

План:
---
{blueprint}
---

Текст:
{chunk}
---
"""

class BlueprintCluster:
    def __init__(self, client):
        self.client = client
        self.pattern = re.compile(r"^\s*\d+[.)]\s*(.*?)(\?|$)", re.M)


    def extract_questions(self, bp: str) -> List[str]:
        """Из плана '1. Q – A' вытягиваем только вопросы."""
        return [m.group(1).strip() for m in self.pattern.finditer(bp)]
    
    
    async def generalize_questions(self, questions: List[str]) -> str:
        questions_str = "\n".join(f"- {q}" for q in questions)
        myprompt = QUESTIONS_PROMPT.format(questions_str=questions_str)
    
        res = await self.client.get_completion(myprompt, max_tokens=64, rep_penalty=1.0)
    
        response = extract_response(res)
        return response
    
    
    async def generate_blueprint(self, chunk):
        myprompt = BLUEPRINT_PROMPT.format(chunk)
    
        blupr = await self.client.get_completion(
            myprompt,
            max_tokens=4000,
            rep_penalty=1.0
        )

        blueprint = extract_response(blupr)
        return blueprint
    
    
    async def summarize_with_blueprint(self, chunk, blueprint):
        myprompt = SUMMARIZE_BLUEPRINT_PROMPT.format(blueprint=blueprint, chunk=chunk)
        sumry = await self.client.get_completion(
            myprompt,
            max_tokens=4000,
            rep_penalty=1.0
        )
    
        summary = extract_response(sumry)
    
        return summary
    
    
    async def build_global_plan(self, chunks: List[str], max_q_per_chunk: int = 50, sample_per_cluster: int = 20) -> List[str]:
        """Генерируем список обобщённых вопросов."""
        # 1) собираем вопросы со всех чанков
        results = AsyncList()
    
        for chunk in chunks:
            results.append(self.generate_blueprint(chunk))
    
        await results.complete_couroutines(batch_size=30)
        blueprints = await results.to_list()
    
        questions = []
    
        for bp in blueprints:
            questions.extend(self.extract_questions(bp)[:max_q_per_chunk])
    
        # 2) K‑means
        emb = encoder.encode(questions, convert_to_tensor=False, normalize_embeddings=True)
        emb = np.asarray(emb, dtype='float32')
    
        k = max(2, int(math.sqrt(len(questions))))
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
    
        clus = faiss.Clustering(dim, k)
        clus.niter = 20
        clus.max_points_per_centroid = 10_000
    
        clus.train(emb, index)
        _, I = index.search(emb, 1)
    
        clusters: Dict[int, List[str]] = {}
    
        for lbl, q in zip(I.ravel().tolist(), questions):
            clusters.setdefault(lbl, []).append(q)
    
        # 3) обобщаем
        gen_tasks_results = AsyncList()
    
        for qs in clusters.values():
            sample = random.sample(qs, min(sample_per_cluster, len(qs)))
            gen_tasks_results.append(self.generalize_questions(sample))
    
        await gen_tasks_results.complete_couroutines(batch_size=30)
        gen_tasks = await gen_tasks_results.to_list()
    
        return gen_tasks
    
    
    async def cluster_text_blueprint_summary(self, chunks, word_limit=500):
        global_plan = await self.build_global_plan(chunks)
        blueprint_glob = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(global_plan))
    
        results = AsyncList()
    
        for chunk in chunks:
            results.append(self.summarize_with_blueprint(chunk, blueprint_glob))
    
        await results.complete_couroutines(batch_size=30)
        summaries = await results.to_list()
    
        while len(summaries) > 1:
            merged_level = []
            i = 0
    
            while i < len(summaries):
                if i + 1 < len(summaries):
                    combo = f"{summaries[i]} {summaries[i + 1]}".strip()
    
                    if len(combo.split()) > word_limit:
                        combo = await self.summarize_with_blueprint(combo, blueprint_glob)
    
                    merged_level.append(combo)
                    i += 2
                else:
                    merged_level.append(summaries[i])
                    i += 1
    
            summaries = merged_level
    
        final_summary = summaries[0].strip()
    
        if len(final_summary.split()) > word_limit:
            for _ in range(3):
                final_summary = await self.summarize_with_blueprint(final_summary, blueprint_glob)
    
                if len(final_summary.split()) <= word_limit:
                    break
    
        return final_summary

    async def run(self, chunks, initial_word_limit=500):
       return await self.cluster_text_blueprint_summary(chunks, initial_word_limit)
