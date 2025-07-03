import json
from blueprint import Blueprint
from cluster_blueprint import BlueprintCluster
from hierarchical import Hierarchical
from iterative import Iterative
from pseudo import Pseudo
from utils import LlmCompleter, chunk_text

class Summarisation:
    def __init__(self, KEY=None, URL=None, model_name=None):
        self.client = LlmCompleter(URL, KEY, model_name)
        self.blueprint = Blueprint(self.client)
        self.cluster_blueprint = Blueprint(self.client, mode='cluster')
        self.hierarchical = Hierarchical(self.client)
        self.iterative = Iterative(self.client)
        self.pseudo = Pseudo(self.client)
        try:
            with open('combined_data.json', 'r', encoding='utf-8') as f:
                self.collection = json.load(f)
        except:
            raise ValueError("No file with annotations and book texts!")