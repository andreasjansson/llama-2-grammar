# Llama 2 grammar

[![Run on Replicate](https://replicate.com/andreasjansson/llama-2-13b-function/badge)](https://replicate.com/andreasjansson/llama-2-13b-grammar)

Llama 2 with grammar-based decoding (provided by llama.cpp).

# Adding new models
Add your .ggml file to https://console.cloud.google.com/storage/browser/replicate-weights/llamacpp

In Makefile, add your model name to line 2:

```
.PHONY: all
all: existing-model-1 existing-model-2 your-new-model
```

Create a destination model at [https://replicate.com/create](replicate.com/create)
At the end of Makefile, add your new model

```
.PHONY: your-new-model
xwin-mlewd-13b-v0-2:
	echo "your_new_model_weights.gguf" > model.txt
	cog push r8.im/your-name/your-new-model
```

Run `cog login` to authenticate your replicate account

Run `sudo make`

N.B. You may need to remove or comment out the other models from the makefile temporarily, as you will not be able to push to andreasjansson's account.