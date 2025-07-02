from utils import AsyncList, extract_response
from metrics import similarity
import torch
import gc
from sentence_transformers import SentenceTransformer

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer('deepvk/USER-bge-m3').to(self.device)
    
    def filter_near_duplicates(self, summaries, th: float = 0.85):
        """сохраняем первую, все следующие сравниваем с последней сохранённой"""
        n = len(summaries)

        if n == 0 or n == 1:
            return summaries
        #print('starting to emb')
        embs = torch.from_numpy(self.encoder.encode(summaries, batch_size=16, normalize_embeddings=True, device=self.device))
        embs = embs.to(self.device)
        sim_matrix = embs @ embs.T
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=0).to(self.device)
        masked_sim = sim_matrix.masked_fill(mask, -1)
        max_sim_row, _ = torch.max(masked_sim, dim=1)
        keep_mask = max_sim_row < th
        keep_mask[0] = True
        
        valid_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze()
        valid_summaries = [summaries[i] for i in valid_indices]
        return valid_summaries
    
    async def summarize_chunk(self, chunk, word_limit=500):
        myprompt = CHUNK_SUMMARY_PROMPT.format(chunk=chunk, word_limit=word_limit)
    
        res = await self.client.get_completion(
            myprompt,
            max_tokens=4000,
            rep_penalty=1.0
        )

        result = extract_response(res)
    
        return result
    
    
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
    
    async def merge_group(self, group1, group2, current_word_limit):
        temp_summary = await self.merge_summaries(group1, current_word_limit)
        temp_summary = await self.merge_summaries(group2, current_word_limit, use_context=True, previous_summary=temp_summary)
        return temp_summary
        
    async def hierarchical_summary(self, chunks, initial_word_limit=500, filtered=True):
        if not chunks:
            raise ValueError("`chunks` должен содержать хотя бы один элемент!")
        #print('chunks len: ', len(chunks))
        rest_chunks = self.filter_near_duplicates(chunks) if filtered else chunks
        if filtered:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
        results = AsyncList()
    
        for chunk in rest_chunks:
            results.append(self.summarize_chunk(chunk, initial_word_limit))
    
        await results.complete_couroutines(batch_size=40)
        summaries = await results.to_list()
        #print('sum len: ', len(summaries))
        current_level_summaries = summaries
        current_word_limit = initial_word_limit
    
        #if len(current_level_summaries) == 0 and filtered:
        #    raise RuntimeError("Не осталось ни одной аннотации после фильтрации узлов!")
    
        if len(current_level_summaries) == 1:
            return current_level_summaries[0]
    
        if len(current_level_summaries) == 2:
            return await self.merge_summaries(current_level_summaries, current_word_limit)

        count = 0
        while len(current_level_summaries) > 2:
            #print('count: ', count)
            count += 1
            i = 0
            tasks = AsyncList()
            while i < len(current_level_summaries):
                if i + 2 < len(current_level_summaries):
                    group1 = current_level_summaries[i: i + 3]
                    if i + 5 < len(current_level_summaries):
                        group2 = current_level_summaries[i + 3: i + 6]
                        tasks.append(self.merge_group(group1, group2, current_word_limit))
                        i += 6
                    else:
                        tasks.append(self.merge_summaries(group1, current_word_limit))
                        i += 3
                else:
                    tasks.append(current_level_summaries[i])
                    i += 1
            #print('waiting...')
            await tasks.complete_couroutines(batch_size=40)
            next_level_summaries = await tasks.to_list()
            current_level_summaries = self.filter_near_duplicates(next_level_summaries) if filtered else next_level_summaries
            if filtered:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
            #print('Done!')
            #print('len of new sum: ', len(current_level_summaries))
    
        if len(current_level_summaries) == 1:
            return current_level_summaries[0]
    
        return await self.merge_summaries(current_level_summaries, current_word_limit)

    async def run(self, chunks, initial_word_limit=500, filtered=True):
       return await self.hierarchical_summary(chunks, initial_word_limit, filtered)
