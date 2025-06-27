from utils import AsyncList, extract_response

BLUEPRINT_PROMPT = """Для следующего отрывка текста создайте план, обязательно состоящий из последовательности вопросов и ответов (не более 15 пар, лучше использовать только ключевые вопросы), которые помогут выделить основные события, персонажей и ключевые моменты. Создавайте только план, не добавляя ничего лишнего. Убедитесь, что каждый вопрос обязательно сопровождается четким и кратким ответом.

Текст:
---
{chunk}
---
"""

SUMMARIZE_BLUEPRINT_PROMPT = """
Используя следующий план из вопросов и ответов, создайте краткое содержание представленного далее текста.
Убедитесь, что текст логически связан и сохраняет важные элементы исходного контекста. Не добавляйте ничего лишнего в ответе.

План:
---
{blueprint}
---

Текст:
---
{chunk}
---
"""

class Blueprint:
    def __init__(self, client):
        self.client = client

    async def generate_blueprint(self, chunk):
        myprompt = BLUEPRINT_PROMPT.format(chunk=chunk)
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
            max_tokens=2000,
            rep_penalty=1.0
        )

        summary = extract_response(sumry)
    
        return summary
    
    
    async def text_blueprint_summary(self, chunks, word_limit=500):
        results = AsyncList()
    
        for chunk in chunks:
            results.append(self.generate_blueprint(chunk))
    
        await results.complete_couroutines(batch_size=30)
        blueprints = await results.to_list()
        
        #summaries = []
        #for chunk, blueprint in zip(chunks, blueprints):
        #    summaries.append(summarize_with_blueprint(model, chunk, blueprint))
    
        summaries_list = AsyncList()
    
        for chunk, blueprint in zip(chunks, blueprints):
            summaries_list.append(self.summarize_with_blueprint(chunk, blueprint))
    
        await summaries.complete_couroutines(batch_size=30)
        summaries = await summaries_list.to_list()
    
        while len(summaries) > 1:
            merged_level = []
            i = 0
    
            while i < len(summaries):
                if i + 1 < len(summaries):
                    combo = f"{summaries[i]} {summaries[i + 1]}".strip()
    
                    if len(combo.split()) > word_limit:
                        bp = await self.generate_blueprint(combo)
                        combo = await self.summarize_with_blueprint(combo, bp)
    
                    merged_level.append(combo)
                    i += 2
                else:
                    merged_level.append(summaries[i])
                    i += 1
    
            summaries = merged_level
    
        final_summary = summaries[0].strip()
    
        if len(final_summary.split()) > word_limit:
            for _ in range(3):
                bp = await self.generate_blueprint(final_summary)
                final_summary = await self.summarize_with_blueprint(final_summary, bp)
    
                if len(final_summary.split()) <= word_limit:
                    break
    
        return final_summary
