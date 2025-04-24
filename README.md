# llm-attack-kit
A collection of LLM attacks, evaluated on the [JailbreakBench](https://jailbreakbench.github.io) benchmark.

## Before you start

1. Setup environmental variables:
```bash
export HF_TOKEN = "YOUR HF TOKEN"

export WANDB_API_KEY = "YOUR WANDB KEY"

# SET ONLY IF YOU NEED CLOUD INFERENCE. BY DEFAULT, OLLAMA INFERENCE IS USED:

export OPENAI_API_KEY = "YOUR OPENAI KEY"

export TOGETHERAI_API_KEY = "YOUR KEY"
```

2. Install [Ollama](https://ollama.com) for local model inference

```bash
!curl -fsSL https://ollama.com/install.sh | sh
```

3. Install required modules (tested on Google Colab Linux instance)

```python
pip3 install -r requirements.txt
```

## Red Teaming TextGrad

The red-teaming implementation of [TextGrad](https://textgrad.com) framework that further tunes the jailbroken prompt using 'textual' gradient descent.

We split the JailbreakBench dataset into train, val and test sets, and run optimization in the usual PyTorch way, changing the system prompt of the target model.

Evaluation metric: Attack Success Rate (ASR).

Run the benchmark:

1. [Not necessary] Modify the Ollama endpoint URL, attacking and target models, num. of epochs and other params in the ```textgrad-redteam/config.py``` file.

2. Run the script
```bash
python3 textgrad-redteam/main.py
```

3. The results are saved in the Weights & Biases logs and locally.


## GCG
JailbreakBench version of [Universal and Transferrable Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043)

We try to run the JailbreakBench benchmark with the same parameters from artifacts, but on the newer model (Llama 3.1 8b).

Run the benchmark:

```bash
python3 gcg/main.py --model "meta-llama/Llama-3.1-8B"
```

Results are saved into *answers.csv* file inside *gcg* folder

## PAIR Ollama
My own fork of [PAIR](https://github.com/pzhao123/PAIR) attack method with extended Ollama models support and fixed JailbreakBench version. 

Reproduces the original JailbreakBench results on the newer model Llama3.1 8b. 

Run the benchmark:

1. Change directory
```bach
cd pair-ollama
```
2. Give run permission
```bach
chmod +x run.sh
```
3. Run the script in prep mode (installs extra dependencies)
```bach
./run.sh --prepare
```
4. Start the script
```bach
./run.sh
```
