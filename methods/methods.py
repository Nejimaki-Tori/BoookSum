import json
from blueprint import Blueprint
from hierarchical import Hierarchical
from iterative import Iterative
from pseudo import Pseudo
from utils import LlmCompleter, chunk_text
from metrics import Evaluater


class Summarisation:
    def __init__(self, KEY=None, URL=None, device=None, encoder=None, model_name=None):
        self.model_name = model_name
        self.device = device
        self.encoder = encoder
        self.think_pass = ' /no_think' if model_name == 'Qwen3-235B-A22B' or model_name == 'RefalMachine/RuadaptQwen3-32B-Instruct-v2' else ''
        self.client = LlmCompleter(URL, KEY, self.model_name)
        self.client_evaluater = LlmCompleter(URL, KEY, 'llama3-70b')
        self.blueprint = Blueprint(self.client, self.device, self.encoder, mode='default', think_pass=self.think_pass)
        self.hierarchical = Hierarchical(self.client, self.device, self.encoder, think_pass=self.think_pass)
        self.evaluater = Evaluater(evaluater=self.client_evaluater, device=self.device, encoder=self.encoder, pre_load=True)
        try:
            with open('combined_data.json', 'r', encoding='utf-8') as f:
                self.collection = json.load(f)
        except:
            raise ValueError("No file with annotations and book texts!")

    def change_model(self, KEY=None, URL=None, model_name=None):
        self.model_name = model_name
        self.think_pass = ' no_think' if model_name == 'Qwen3-235B-A22B' or model_name == 'RefalMachine/RuadaptQwen3-32B-Instruct-v2' else ''
        self.client = LlmCompleter(URL, KEY, self.model_name)
        self.blueprint = Blueprint(self.client, self.device, self.encoder, mode='default', think_pass=self.think_pass)
        self.hierarchical = Hierarchical(self.client, self.device, self.encoder, think_pass=self.think_pass)