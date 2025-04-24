from huggingface_hub import login
import os
import pandas as pd
import nanogcg
import torch
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(HF_TOKEN)
    print("Successfully logged in to Hugging Face!")
else:
    raise ValueError(
        "Hugging Face token not found. Please set the HF_TOKEN environment variable."
    )

# dataset
splits = {
    "harmful": "data/harmful-behaviors.csv",
    "benign": "data/benign-behaviors.csv",
}
df = pd.read_csv("hf://datasets/JailbreakBench/JBB-Behaviors/" + splits["harmful"])

answers = []
prompts = []


model_id = "meta-llama/Llama-3.1-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

for goal, target in zip(df["Goal"], df["Target"]):
    # params taken from JailbrekBench artifacts
    config = GCGConfig(
        num_steps=500,
        search_width=50,
        batch_size=512,
        topk=256,
        seed=42,
        verbosity="WARNING",
    )

    result = nanogcg.run(model, tokenizer, goal, target, config)
    prompt = result.best_string
    prompts.append(prompt)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    generated_tokens_ids = outputs[0]
    answer = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True)
    answers.append(answer)


pd.DataFrame.from_dict(
    {"goal": df["Goal"], "prompt": prompts, "answer": answers}
).to_csv(f"{model_id}_answers.csv")
