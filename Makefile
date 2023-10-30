.PHONY: all
all: codellama-34b-instruct codellama-7b-instruct llama-2-70b-chat llama-2-13b-chat llama-2-13b wizardcoder-python-34b-v1 xwin-mlewd-13b-v0-2

.PHONY: codellama-34b-instruct
codellama-34b-instruct:
	echo "codellama-34b-instruct.Q5_K_S.gguf" > model.txt
	cog push r8.im/andreasjansson/codellama-34b-instruct-gguf

.PHONY: codellama-7b-instruct
codellama-7b-instruct:
	echo "codellama-7b-instruct.Q5_K_S.gguf" > model.txt
	cog push r8.im/andreasjansson/codellama-7b-instruct-gguf

.PHONY: llama-2-70b-chat
llama-2-70b-chat:
	echo "llama-2-70b-chat.Q4_K_M.gguf" > model.txt
	cog push r8.im/andreasjansson/llama-2-70b-chat-gguf

.PHONY: llama-2-13b-chat
llama-2-13b-chat:
	echo "llama-2-13b-chat.Q5_K_S.gguf" > model.txt
	cog push r8.im/andreasjansson/llama-2-13b-chat-gguf

.PHONY: llama-2-13b
llama-2-13b:
	echo "llama-2-13b.Q5_K_S.gguf" > model.txt
	cog push r8.im/andreasjansson/llama-2-13b-gguf

.PHONY: wizardcoder-python-34b-v1
wizardcoder-python-34b-v1:
	echo "wizardcoder-python-34b-v1.0.Q5_K_M.gguf" > model.txt
	cog push r8.im/andreasjansson/wizardcoder-python-34b-v1-gguf

.PHONY: xwin-mlewd-13b-v0-2
xwin-mlewd-13b-v0-2:
	echo "Xwin-MLewd-13B-V0.2.q5_k_m.gguf" > model.txt
	cog push r8.im/andreasjansson/xwin-mlewd-13b-v0-2
