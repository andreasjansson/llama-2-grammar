import random
from scipy.io import wavfile
import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
import pprint as pp
from llama_cpp import LlamaGrammar, Llama


examples = """
samples=t*(((t>>12)|(t>>8))&(63&(t>>4)))
samples=(t*(t>>5|t>>8))>>(t>>16)
samples=t*(((t>>9)|(t>>13))&(25&(t>>6)))
samples=t*(((t>>11)&(t>>8))&(123&(t>>3)))
samples=t*(t>>8*((t>>15)|(t>>8))&(20|(t>>19)*5>>t|(t>>3)))
samples=(-t&4095)*(255&t*(t&(t>>13)))>>12)+(127&t*(234&t>>8&t>>3)>>(3&t>>14))
samples=t*(t>>((t>>9)|(t>>8))&(63&(t>>4)))
samples=(t>>6|t|t>>(t>>16))*10+((t>>11)&7)
samples=(t|(t>>9|t>>7))*t&(t>>11|t>>9)
samples=t*5&(t>>7)|t*3&(t*4>>10)
samples=(t>>7|t|t>>6)*10+4*(t&t>>13|t>>6)
samples=((t&4096)?((t*(t^t%255)|(t>>4))>>1):(t>>3)|((t&8192)?t<<2:t))
samples=((t*(t>>8|t>>9)&46&t>>8))^(t&t>>13|t>>6)
samples=(t*5&t>>7)|(t*3&t>>10)
samples=(t&t%255)-(t*3&t>>13&t>>6)
samples=t>>4|t&((t>>5)/(t>>7-(t>>15)&-t>>7-(t>>15)))
samples=(t*9&t>>4|t*5&t>>7|t*3&t/1024)-1
samples=((t*(t>>12)&(201*t/100)&(199*t/100))&(t*(t>>14)&(t*301/100)&(t*399/100)))+((t*(t>>16)&(t*202/100)&(t*198/100))-(t*(t>>17)&(t*302/100)&(t*298/100)))
samples=((t*(t>>12)&(201*t/100)&(199*t/100))&(t*(t>>14)&(t*301/100)&(t*399/100)))+((t*(t>>16)&(t*202/100)&(t*198/100))-(t*(t>>18)&(t*302/100)&(t*298/100)))
samples=t*(t^t+(t>>15|1)^(t-1280^t)>>10)
samples=((t>>1%128)+20)*3*t>>14*t>>18
samples=t*(((t>>9)&10)|((t>>11)&24)^((t>>10)&15&(t>>15)))
samples=t
samples=t&t>>8
samples=t*(42&t>>10)
samples=t|t%255|t%257
samples=t>>6&1?t>>5:-t>>4
samples=t*(t>>9|t>>13)&16
samples=(t&t>>12)*(t>>4|t>>8)
samples=(t*5&t>>7)|(t*3&t>>10)
samples=(t*(t>>5|t>>8))>>(t>>16)
samples=t*5&(t>>7)|t*3&(t*4>>10)
samples=(t>>13|t%24)&(t>>7|t%19)
samples=(t*((t>>9|t>>13)&15))&129
samples=(t&t%255)-(t*3&t>>13&t>>6)
samples=(t&t>>12)*(t>>4|t>>8)^t>>6
samples=t*(((t>>9)^((t>>9)-1)^1)%13)
samples=(t/8)>>(t>>9)*t/((t>>14&3)+4)
samples=(~t/100|(t*3))^(t*3&(t>>5))&t
samples=(t|(t>>9|t>>7))*t&(t>>11|t>>9)
samples=((t>>1%128)+20)*3*t>>14*t>>18
samples=t*(((t>>12)|(t>>8))&(63&(t>>4)))
samples=t*(((t>>9)|(t>>13))&(25&(t>>6)))
samples=t*(t^t+(t>>15|1)^(t-1280^t)>>10)
samples=t*(((t>>11)&(t>>8))&(123&(t>>3)))
samples=(t>>7|t|t>>6)*10+4*(t&t>>13|t>>6)
samples=(t*9&t>>4|t*5&t>>7|t*3&t/1024)-1
samples=t*(t>>((t>>9)|(t>>8))&(63&(t>>4)))
samples=(t>>6|t|t>>(t>>16))*10+((t>>11)&7)
samples=(t>>(t&7))|(t<<(t&42))|(t>>7)|(t<<5)
samples=(t>>7|t%45)&(t>>8|t%35)&(t>>11|t%20)
samples=(t>>6|t<<1)+(t>>5|t<<3|t>>3)|t>>2|t<<1
samples=t+(t&t^t>>6)-t*((t>>9)&(t%16?2:6)&t>>9)
samples=((t*(t>>8|t>>9)&46&t>>8))^(t&t>>13|t>>6)
samples=(t>>5)|(t<<4)|((t&1023)^1981)|((t-67)>>4)
samples=t>>4|t&(t>>5)/(t>>7-(t>>15)&-t>>7-(t>>15))
samples=t*(t/256)-t*(t/255)+t*(t>>5|t>>6|t<<2&t>>1)
"""


class SynthOutput(BaseModel):
    sample_expression: str
    audio: Path


class Predictor(BasePredictor):
    def setup(self):
        self.grammar = LlamaGrammar.from_string(
            """
root   ::= "sample=" expr
expr   ::= term | term infix term | prefix term
term   ::= "t" | num | "(" expr ")"
num    ::= [0-9] | [0-9][0-9] | [0-9][0-9][0-9] | [0-9][0-9][0-9][0-9]
prefix ::= "-" | "~"
infix  ::= "&" | "|" | "+" | "-" | "*" | "/" | ">>" | "<<" | "%" | "^"
"""
        )
        model_path = "/models/codellama-7b.Q5_K_S.gguf"
        self.llm = Llama(model_path, n_ctx=2048, n_gpu_layers=-1, main_gpu=0, n_threads=1)
        #self.llm = Llama(model_path, n_ctx=2048)

    def predict(
        self,
        duration: int = Input(description="Duration in seconds", default=5),
        sample_expression: str = Input(
            description="You can provide a sample expression in the format sample=... if you only want to use the sytnehsizer",
            default=None,
        ),
    ) -> SynthOutput:
        if not sample_expression:
            output = self.llm(
                "# Examples of one-liners that generate code" + examples,
                grammar=self.grammar,
                max_tokens=100,
            )
            sample_expression = output["choices"][0]["text"]
        fun = eval(sample_expression.replace("sample=", "lambda t: "))

        def apply(fun):
            out = []
            for t in range(8000 * duration):
                x = int(fun(t)) % 127
                out.append(x)
            return out

        x = apply(fun)
        scaled = np.int16(x / np.max(np.abs(x)) * 32767)

        out_path = Path("/tmp/out.wav")
        wavfile.write(str(out_path), 8000, scaled)

        return SynthOutput(sample_expression=sample_expression, audio=out_path)
