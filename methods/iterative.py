from utils import extract_response

INITIAL_SUMMARY_PROMPT = """Ниже приведена начальная часть истории:
---
{chunk}
---

Мы последовательно проходим по фрагментам истории, чтобы постепенно обновить общее описание всего сюжета. Напишите краткое содержание для приведенного выше отрывка, не забудьте включить важную информацию, относящуюся к ключевым событиям, предыстории, обстановке, персонажам, их целям и мотивам. Вы должны кратко представить персонажей, места и другие важные элементы, если они упоминаются в аннотации впервые. История может содержать нелинейные повествования, ретроспективные кадры, переключение между альтернативными мирами или точками зрения и т.д. Поэтому вам следует организовать аннотацию таким образом, чтобы она представляло собой последовательное и хронологическое изложение. Несмотря на этот пошаговый процесс обновления аннотации, вам необходимо создать аннотацию, которая будет выглядеть так, как будто оно написано на одном дыхании. Краткое содержание должно содержать примерно {word_limit} слов и может состоять из нескольких абзацев. Пиши на русском языке без грамматических ошибок, пожалуйста.

Краткое содержание:
"""

INTERMEDIATE_SUMMARY_PROMPT = """Ниже приведен фрагмент из истории:
---
{chunk}
---
Ниже приводится краткое изложение истории, произошедшей до этого момента:
---
{current_summary}
---
Мы последовательно просматриваем фрагменты истории, чтобы постепенно обновить общее описание всего сюжета. Вам необходимо обновить краткое содержание, чтобы включить любую новую важную информацию в текущий отрывок. Эта информация может касаться ключевых событий, предыстории, обстановки, персонажей, их целей и мотивов. Вы должны кратко представить персонажей, места и другие важные элементы, если они впервые упоминаются в аннотации. История может содержать нелинейные повествования, ретроспективные кадры, переключение между альтернативными мирами или точками зрения и т.д. Поэтому вам следует организовать аннотацию таким образом, чтобы она представляла собой последовательное и хронологическое изложение. Несмотря на этот пошаговый процесс обновления аннотацию, вам необходимо создать аннотацию, которая будет выглядеть так, как будто оно написано на одном дыхании. Обновленная аннотация должно содержать примерно {word_limit} слов и может состоять из нескольких абзацев. Пиши на русском языке без грамматических ошибок, пожалуйста.

Обновленная аннотация:
"""

COMPRESS_SUMMARY_PROMPT = """Ниже приводится краткое изложение части истории:
---
{current_summary}
---
В настоящее время это краткое изложение содержит {current_length} слов. Ваша задача - сократить его до {target_length} слов. Краткое изложение должно оставаться ясным, всеобъемлющим и плавным, но при этом быть кратким. По возможности, сохраняйте подробности о ключевых событиях, предыстории, обстановке, персонажах, их целях и мотивах, но излагайте эти элементы более кратко. Обязательно дайте краткое представление о персонажах, местах и других основных компонентах при первом упоминании в кратком изложении. Удалите незначительные детали, которые мало что добавляют к общей сюжетной линии. В истории могут быть нелинейные сюжеты, ретроспективные кадры, переходы между альтернативными мирами или точками зрения и т.д. Поэтому вам следует составить краткое содержание таким образом, чтобы оно представляло собой последовательное и хронологическое повествование. Пиши на русском языке без грамматических ошибок, пожалуйста.

Краткое содержание:
"""

class Iterative:
    def __init__(self, client):
        self.client = client

    async def generate_initial_summary(self, chunk, word_limit=500):
        myprompt = INITIAL_SUMMARY_PROMPT.format(chunk=chunk, word_limit=word_limit)
    
        res = await self.client.get_completion(myprompt, max_tokens=4000)
    
        result = extract_response(res)
    
        return result
    
    
    async def generate_intermediate_summary(self, chunk, current_summary, word_limit=500):
        myprompt = INTERMEDIATE_SUMMARY_PROMPT.format(chunk=chunk, current_summary=current_summary, word_limit=word_limit)
    
        res = await self.client.get_completion(myprompt, max_tokens=4000)
        
        result = extract_response(res)
        
        return result
    
    
    async def compress_summary(self, current_summary, current_length, target_length):
        myprompt = COMPRESS_SUMMARY_PROMPT.format(current_summary=current_summary, current_length=current_length, target_length=target_length)
    
        res = await self.client.get_completion(myprompt, max_tokens=4000)
        
        result = extract_response(res)
        
        return result
    
    
    async def iterative_summary(self, chunks, word_limit=500):
        current_summary = await self.generate_initial_summary(chunks[0], word_limit)
        
        for i in range(1, len(chunks)):
            current_summary = await self.generate_intermediate_summary(chunks[i], current_summary)
            current_length = len(current_summary.split())
    
            if current_length > word_limit:
                current_summary = await self.compress_summary(current_summary, current_length, 500)
    
        return current_summary
        
    async def run(self, chunks, initial_word_limit=500):
       return await self.iterative_summary(chunks, initial_word_limit)
