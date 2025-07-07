import json
from blueprint import Blueprint
from hierarchical import Hierarchical
from iterative import Iterative
from pseudo import Pseudo
from utils import LlmCompleter, chunk_text
from metrics import Evaluater

class Summarisation:
    def __init__(self, KEY=None, URL=None, model_name=None):
        self.model_name = model_name
        self.client = LlmCompleter(URL, KEY, self.model_name)
        self.client_evaluater = LlmCompleter(URL, KEY, 'llama3-70b')
        self.blueprint = Blueprint(self.client)
        self.cluster_blueprint = Blueprint(self.client, mode='cluster')
        self.hierarchical = Hierarchical(self.client)
        self.iterative = Iterative(self.client)
        self.pseudo = Pseudo(self.client)
        self.evaluater = Evaluater(evaluater=self.client_evaluater, pre_load=True)
        try:
            with open('combined_data.json', 'r', encoding='utf-8') as f:
                self.collection = json.load(f)
        except:
            raise ValueError("No file with annotations and book texts!")

    def change_model(self, KEY=None, URL=None, model_name=None):
        self.model_name = model_name
        self.client = LlmCompleter(URL, KEY, self.model_name)