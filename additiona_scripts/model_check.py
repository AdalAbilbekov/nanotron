import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import LlamaConfig as HFLlamaConfig

TEST_PROMPT = "Сәлем"

def check_converted_model_generation(save_path: Path):
    """Loads a huggingface model and tokenizer from `save_path` and
    performs a dummy text generation."""

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    input_ids = tokenizer(TEST_PROMPT, return_tensors="pt")["input_ids"].cuda()
    print("Inputs:", tokenizer.batch_decode(input_ids))

    model = LlamaForCausalLM.from_pretrained(save_path).cuda().bfloat16()
    out = model.generate(input_ids, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id, temperature=0.2, top_p=1.0, repetition_penalty=2.0)
    
    print("Generation (converted): ", tokenizer.batch_decode(out))

if __name__=="__main__":
    check_converted_model_generation("/scratch/adal_abilbekov/models/Llama_1.5B_10-02-2025/42000_hf")