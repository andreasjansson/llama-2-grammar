import json
import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
import pprint as pp
from llama_cpp import LlamaGrammar, Llama

from json_schema_to_grammar import SchemaConverter


class Predictor(BasePredictor):
    def setup(self):
        model_path = "/models/llama-2-13b.Q5_K_S.gguf"
        self.llm = Llama(
            model_path, n_ctx=2048, n_gpu_layers=-1, main_gpu=0, n_threads=1
        )

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        jsonschema: str = Input(description="JSON schema for the generated output"),
        max_tokens: int = Input(
            description="Max number of tokens to return", default=500
        ),
    ) -> str:
        prompt = (
            prompt
            + f"""

Respond with json that adheres to the following jsonschema:

{jsonschema}
"""
        )

        schema = json.loads(jsonschema)
        converter = SchemaConverter({})
        converter.visit(schema, "")
        grammar = LlamaGrammar.from_string(converter.format_grammar())

        output = self.llm(
            prompt,
            grammar=grammar,
            max_tokens=max_tokens,
        )["choices"][
            0
        ]["text"]
        print(output)
        return json.dumps(json.loads(output), indent=2)
