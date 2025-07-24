from typing import List
from evaluate import load as hf_load
from rouge import Rouge
from utils import AsyncList, model, extract_response
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import razdel

QUESTIONS_COVERAGE_PROMPT = """Ты - эксперт в оценивании качества аннотаций для книг. Твоя задача — тщательно оценить, насколько представленная аннотация позволяет ответить на конкретный вопрос, касающийся ключевых аспектов исходного произведения.

Вопрос:
{question}
Текст аннотации:
{text}

Содержится ли в этом тексте ответ на вопрос?
Начни ответ с {yes}, если содержится или с {no}, если не содержится.
"""

GOLD_QUESTIONS_PROMPT = """На основе данной аннотации сформируй несколько ключевых вопросов, ответы на которые можно однозначно дать, зная содержание аннотации.

Аннотация:
---
{ref_annotation}
---
Ключевые вопросы нужно писать по порядку, начиная каждый вопрос с новой строки. Кроме ключевых вопросов ничего писать не нужно.
"""

ANSWER_PROMPT = """На основе данной аннотации сформируй ответ на поставленный вопрос. Твоя задача — **строго на основе предоставленной аннотации** сформировать ответ на ключевой вопрос.

Аннотация:
{ref_annotation}

Ключевой вопрос:
{key_q}
---

Не пиши вопрос, пиши только ответ.
"""

class Evaluater:
    def __init__(self, evaluater=None, device=None, encoder=None, pre_load=False):
        if not pre_load:
            nltk.download('punkt')
            nltk.download('punkt_tab')
        self.client_eval = evaluater
        self.device = device
        self.encoder = encoder
        self.stemmer = SnowballStemmer('russian')
        #self.rouge = hf_load('rouge')
        self.bert_score = hf_load('bertscore', model_type='deepvk/USER-bge-m3')
        
    def lemmatize_text(self, text):
        tokens = word_tokenize(text, language='russian')
        return ' '.join(self.stemmer.stem(token) for token in tokens)

    def clean(self, txt):
        txt = re.sub(r"<think>.*?</think>", " ", txt, flags=re.DOTALL)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    
    def rouge_L(self, ref, pred):
        rouge = Rouge()

        ref = self.clean(ref)
        pred = self.clean(pred)

        scores = rouge.get_scores(
            hyps=[pred],  
            refs=[ref],   
            avg=False               
        )
        rouge_l = scores[0]["rouge-l"]
        return rouge_l['f']

    #def bert_f1(self, ref, pred):
    #    return self.bert_score.compute(references=[ref], predictions=[pred], lang='ru')['f1'][0]

    def bertscore(self, ref, pred):
        ref = self.clean(ref)
        pred = self.clean(pred)
        ref_emb =  self.encoder.encode([s.text for s in razdel.sentenize(ref)], normalize_embeddings=True, device=self.device)
        pred_emb = self.encoder.encode([s.text for s in razdel.sentenize(pred)], normalize_embeddings=True, device=self.device)
        sims = pred_emb @ ref_emb.T
        precision = sims.max(axis=1).mean()
        recall = sims.max(axis=0).mean()
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
        
    def similarity(self, a, b):
        emb_1 = self.encoder.encode(a, device=self.device)
        emb_2 = self.encoder.encode(b, device=self.device)
    
        return round(float(self.encoder.similarity(emb_1, emb_2).item()), 3)


    async def compute_coverage(
            self,
            questions,
            summary,
            positive_choice="YES",
            negative_choice="NO"
    ):
        probs = AsyncList()
    
        for q in questions:
            myprompt = QUESTIONS_COVERAGE_PROMPT.format(question=q, text=summary, yes=positive_choice, no=negative_choice)
            probs.append(self.client_eval.get_probability(myprompt, rep_penalty=1.0, max_tokens=10))
    
        await probs.complete_couroutines(batch_size=40)
        results = await probs.to_list()

        flags = []
    
        for res in results:
            if negative_choice in res:
                prob = 1 - res[negative_choice]
            elif positive_choice in res:
                prob = res[positive_choice]
            else:
                prob = 0.0
                
            flags.append(1 if prob >= 0.75 else 0)
    
        coverage = sum(flags) / len(flags) if flags else 0.0
    
        return coverage, flags
    
    
    async def generate_key_questions(self, ref_annotation):
        myprompt = GOLD_QUESTIONS_PROMPT.format(ref_annotation=ref_annotation)
        
        res = await self.client_eval.get_completion(myprompt, max_tokens=512)
        result = extract_response(res)

        questions = [q for q in result.split('\n') if q.strip()]
        
        return questions
    
    
    async def get_answer(self, ref_annotation, key_question):
        myprompt = ANSWER_PROMPT.format(ref_annotation=ref_annotation, key_q=key_question)
        res = await self.client_eval.get_completion(myprompt, max_tokens=512)

        answer = extract_response(res)
    
        return answer

    async def generate_answers(self, ref_annotation, questions):
        answers = AsyncList()

        for question in questions:
            answers.append(self.get_answer(ref_annotation, question))

        await answers.complete_couroutines(batch_size=40)
        answers = await answers.to_list()

        return answers
    
    
    def compute_answer_similarity(self, questions, cov_flags, answers_gold, answers_gen):
        sims = []
    
        for flag, gen, gold in zip(cov_flags, answers_gen, answers_gold):
            if flag == 0:
                sims.append(0.0)
            else:
                sims.append(self.similarity(gen, gold))
    
        return sum(sims) / len(sims) if sims else 0.0

    async def compute_similarity(self, ref_annotation, gen_annotation):
        questions = await self.generate_key_questions(ref_annotation)
        #print(questions)
        #print()
        #print()
        answers_gold = await self.generate_answers(ref_annotation, questions)
        #print(answers_gold)
        #print()
        #print()
        answers_gen = await self.generate_answers(gen_annotation, questions)
        #print(answers_gen)
        #print()
        #print()
        coverage, cov_flags = await self.compute_coverage(questions, gen_annotation)
        #print(cov_flags)
        answer_similarity = self.compute_answer_similarity(questions, cov_flags, answers_gold, answers_gen)

        return coverage, answer_similarity

    async def evaluate_annotation(self, ref_annotation, gen_annotation):
        bertscore = self.bertscore(ref_annotation, gen_annotation)
        rouge = self.rouge_L(ref_annotation, gen_annotation)
        coverage, answer_sim = await self.compute_similarity(ref_annotation, gen_annotation)

        return bertscore, rouge, coverage, answer_sim
        