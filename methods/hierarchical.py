from utils import AsyncList, extract_response
from metrics import similarity

CHUNK_SUMMARY_PROMPT = """Ниже приведена часть истории:
---
{chunk}
---

Мы создаем единую всеобъемлющую аннотацию для истории, рекурсивно объединяя фрагменты. Теперь напишите краткое содержание для приведенного выше отрывка, не забудьте включить важную информацию, относящуюся к ключевым событиям, предыстории, обстановке, персонажам, их целям и мотивам. Вы должны кратко представить персонажей, места и другие важные элементы, если они упоминаются в аннотации впервые. История может содержать нелинейные повествования, ретроспективные кадры, переключение между альтернативными мирами или точками зрения и т.д. Поэтому вам следует организовать резюме таким образом, чтобы оно представляло собой последовательное и хронологическое изложение. Несмотря на этот рекурсивный процесс объединения, вам необходимо создать аннотацию, которая будет выглядеть так, как будто она написана на одном дыхании. 

Аннотация должна состоять из не более {word_limit} слов и может включать несколько абзацев.
"""

SUMMARY_MERGE_WITH_CONTEXT_PROMPT = """Ниже приведено краткое изложение контекста, предшествующего некоторым частям истории:
---
{previous_summary}
---

Ниже приведены несколько кратких изложений последовательных частей рассказа:
---
{combined_summary}
---

Мы создаем единую всеобъемлющую аннотацию для истории, рекурсивно объединяя краткие сведения из ее фрагментов. Теперь объедините предыдущий контекст и краткие содержания в одно краткое содержание, не забудьте включить важную информацию, относящуюся к ключевым событиям, фону, обстановке, персонажам, их целям и мотивам. Вы должны кратко представить персонажей, места и другие важные элементы, если они упоминаются в аннотации впервые. История может содержать нелинейные повествования, ретроспективные кадры, переключение между альтернативными мирами или точками зрения и т.д. Поэтому вам следует организовать аннотацию таким образом, чтобы она представляло собой последовательное и хронологическое изложение. Несмотря на этот рекурсивный процесс объединения, вам необходимо создать аннотацию, которая будет выглядеть так, как будто она написана на одном дыхании. Аннотация должна состоять из {word_limit} слов и может включать несколько абзацев.
"""

SUMMARY_MERGE_NO_CONTEXT_PROMPT = """Ниже приведены несколько кратких изложений последовательных частей рассказа:
---
{combined_summary}
---

Мы последовательно проходим по фрагментам истории, чтобы постепенно обновить общее описание всего сюжета. Напишите краткое содержание для приведенного выше отрывка, не забудьте включить важную информацию, относящуюся к ключевым событиям, предыстории, обстановке, персонажам, их целям и мотивам. Вы должны кратко представить персонажей, места и другие важные элементы, если они упоминаются в аннотации впервые. История может содержать нелинейные повествования, ретроспективные кадры, переключение между альтернативными мирами или точками зрения и т.д. Поэтому вам следует организовать аннотацию таким образом, чтобы она представляла собой последовательное и хронологическое изложение. Несмотря на этот пошаговый процесс обновления аннотации, вам необходимо создать аннотацию, которая будет выглядеть так, как будто она написана на одном дыхании. Аннотация должна содержать примерно {word_limit} слов и может состоять из нескольких абзацев.
"""

class Hierarchical:
    def __init__(self, client):
        self.client = client
    
    def filter_near_duplicates(self, summaries, th: float = .85):
        """сохраняем первую, все следующие сравниваем с последней сохранённой"""
        if not summaries:
            return []
    
        kept = [summaries[0]]
    
        for s in summaries[1:]:
            if similarity(s, kept[-1]) < th:
                kept.append(s)
    
        return kept
    
    
    async def summarize_chunk(self, chunk, word_limit=500):
        myprompt = CHUNK_SUMMARY_PROMPT.format(chunk=chunk, word_limit=word_limit)
    
        res = await self.client.get_completion(
            myprompt,
            max_tokens=4000,
            rep_penalty=1.0
        )
    
        return res
    
    
    async def merge_summaries(self, summaries, word_limit=500, use_context=False, previous_summary=''):
        combined_summary = " ".join(summaries)
    
        if len(combined_summary.split()) > word_limit:
            combined_summary = await self.summarize_chunk(combined_summary, word_limit)
    
        if use_context:
            myprompt = SUMMARY_MERGE_WITH_CONTEXT_PROMPT.format(previous_summary=previous_summary, combined_summary=combined_summary, word_limit=word_limit)
        else:
            myprompt = SUMMARY_MERGE_NO_CONTEXT_PROMPT.format(combined_summary=combined_summary, word_limit=word_limit)
    
        res = await self.client.get_completion(
            myprompt,
            max_tokens=4000,
            rep_penalty=1.0
        )
        result = extract_response(res)
    
        return result
    
    
    async def hierarchical_summary(self, chunks, initial_word_limit=500, filtered=True):
        if not chunks:
            raise ValueError("`chunks` должен содержать хотя бы один элемент!")
        
        rest_chunks = self.filter_near_duplicates(chunks) if filtered else chunks
    
        results = AsyncList()
    
        for chunk in rest_chunks:
            results.append(self.summarize_chunk(chunk, initial_word_limit))
    
        await results.complete_couroutines(batch_size=20)
        summaries = await results.to_list()
    
        current_level_summaries = summaries
        current_word_limit = initial_word_limit
    
        if len(current_level_summaries) == 0 and filtered:
            raise RuntimeError("Не осталось ни одной аннотации после фильтрации узлов!")
    
        if len(current_level_summaries) == 1:
            return current_level_summaries[0]
    
        if len(current_level_summaries) == 2:
            return await self.merge_summaries(current_level_summaries, current_word_limit)
    
        while len(current_level_summaries) > 2:
            next_level_summaries = []
            i = 0
    
            while i < len(current_level_summaries):
                if i + 2 < len(current_level_summaries):
                    temp_summary = await self.merge_summaries(current_level_summaries[i: i + 3], current_word_limit)
    
                    if i + 5 < len(current_level_summaries):
                        temp_summary = await self.merge_summaries(current_level_summaries[i + 3: i + 6], current_word_limit, use_context=True, previous_summary=temp_summary)
                        i += 6
                    else:
                        i += 3
    
                    next_level_summaries.append(temp_summary)
                else:
                    next_level_summaries.append(current_level_summaries[i])
                    i += 1
    
            current_level_summaries = self.filter_near_duplicates(next_level_summaries) if filtered else next_level_summaries
    
        if len(current_level_summaries) == 1:
            return current_level_summaries[0]
    
        return await self.merge_summaries(current_level_summaries, current_word_limit)
