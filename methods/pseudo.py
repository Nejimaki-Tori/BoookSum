from utils import AsyncList, extract_response

PSEUDO_PROMPT = """Расскажи, пожалуйста, о чём кника *{name}* не опираясь на текст самой книги."""

class Pseudo:
    def __init__(self, client):
        self.client = client

        
    async def generate_pseudo(self, name):
        myprompt = PSEUDO_PROMPT.format(name=name)
    
        res = await self.client.get_completion(
            myprompt,
            max_tokens=4000,
            rep_penalty=1.0
        )

        result = extract_response(res)
    
        return result
    
    
    async def pseudo_summaries(self, file_names):
        results = AsyncList()
    
        for name in file_names:
            results.append(self.generate_pseudo(name))
    
        await results.complete_couroutines(batch_size=40)
        summaries = await results.to_list()
    
        summaries_with_names = []
    
        for name, summ in zip(file_names, summaries):
            summaries_with_names.append((name, summ))
    
        return summaries_with_names

    async def run(self, file_names):
        return await pseudo_summaries(file_names)
