import time
import subprocess
import json
import numpy as np
from cog import BasePredictor, Input, Path, BaseModel, ConcatenateIterator
import pprint as pp
from llama_cpp import LlamaGrammar, Llama


class Predictor(BasePredictor):
    def setup(self):
        model_path = "/models/model.gguf"
        model_url = "https://storage.googleapis.com/replicate-weights/llamacpp/llama-2-13b.Q5_K_S.gguf"
        print("Downloading model weights....")
        start = time.time()
        subprocess.check_call(["pget", model_url, model_path])
        print("Downloading weights took: ", time.time() - start)
        self.llm = Llama(
            model_path, n_ctx=2048, n_gpu_layers=-1, main_gpu=0, n_threads=1
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        grammar: str = Input(description="Grammar in GBNF format"),
        max_tokens: int = Input(
            description="Max number of tokens to return", default=500
        ),
    ) -> ConcatenateIterator[str]:
        grammar = LlamaGrammar.from_string(grammar)

        for tok in self.llm(
            prompt,
            grammar=grammar,
            max_tokens=max_tokens,
            stream=True,
        ):
            yield tok['choices'][0]['text']
