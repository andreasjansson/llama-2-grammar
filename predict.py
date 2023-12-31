import os
import time
import subprocess
import json
import numpy as np
from cog import BasePredictor, Input, Path, BaseModel, ConcatenateIterator
import pprint as pp
from llama_cpp import LlamaGrammar, Llama

from json_schema_to_grammar import SchemaConverter


class Predictor(BasePredictor):
    def setup(self):
        # model.txt is generated by the Makefile
        with open("model.txt") as f:
            model = f.read().strip()
        model_path = f"/models/{model}"
        model_url = f"https://storage.googleapis.com/replicate-weights/llamacpp/{model}"
        start = time.time()
        if not os.path.exists(model_path):
            print("Downloading model weights....")
            subprocess.check_call(["pget", model_url, model_path])
            print("Downloading weights took: ", time.time() - start)
        self.llm = Llama(
            model_path, n_ctx=2048, n_gpu_layers=-1, main_gpu=0, n_threads=1
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        grammar: str = Input(
            description="Grammar in GBNF format. Use either grammar or jsonschema.",
            default=None,
        ),
        jsonschema: str = Input(
            description="JSON schema for the generated output. Use either grammar or jsonschema. You can use the jsonschema in the prompt by using the special string '{jsonschema}'",
            default=None,
        ),
        max_tokens: int = Input(
            description="Max number of tokens to return", default=500
        ),
        temperature: float = Input(description="Temperature", default=0.8),
        top_p: float = Input(description="Top P", default=0.95),
        top_k: int = Input(description="Top K", default=10),
        frequency_penalty: float = Input(
            description="Frequency penalty", ge=0.0, le=2.0, default=0.0
        ),
        presence_penalty: float = Input(
            description="Presence ", ge=0.0, le=2.0, default=0.0
        ),
        repeat_penalty: float = Input(
            description="Repetition penalty", ge=0.0, le=2.0, default=1.1
        ),
        mirostat_mode: str = Input(
            description="Mirostat sampling mode",
            choices=["Disabled", "Mirostat", "Mirostat 2.0"],
            default="Disabled",
        ),
        mirostat_learning_rate: float = Input(
            description="Mirostat learning rate, if mirostat_mode is not Disabled",
            ge=0,
            le=1,
            default=0.1,
        ),
        mirostat_entropy: float = Input(
            description="Mirostat target entropy", ge=0, le=10, default=5.0
        ),
    ) -> ConcatenateIterator[str]:
        if grammar and jsonschema:
            raise ValueError("Use either grammar or jsonschema, not both.")

        if jsonschema:
            if "{jsonschema}" in prompt:
                prompt = prompt.replace("{jsonschema}", jsonschema)
            schema = json.loads(jsonschema)
            converter = SchemaConverter({})
            converter.visit(schema, "")
            grammar = converter.format_grammar()

        if grammar and grammar.strip():
            grammar = LlamaGrammar.from_string(grammar)
        else:
            grammar = None

        print("Prompt:\n" + prompt)

        for tok in self.llm(
            prompt,
            grammar=grammar,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            mirostat_mode={"Disabled": 0, "Mirostat": 1, "Mirostat 2.0": 2}[
                mirostat_mode
            ],
            mirostat_eta=mirostat_learning_rate,
            mirostat_tau=mirostat_entropy,
        ):
            yield tok["choices"][0]["text"]
