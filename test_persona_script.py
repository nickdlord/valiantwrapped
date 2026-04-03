#!/usr/bin/env python3
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from author_persona_llama31 import (
    MODEL_ID,
    make_persona_messages,
    generate_chat,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author-label", default="Test_Author_12345")
    ap.add_argument("--themes", required=True, help="Themes text")
    ap.add_argument("--summary", required=True, help="Summary text")
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )

    messages = make_persona_messages(
        author_label=args.author_label,
        themes=args.themes,
        summary=args.summary,
    )

    print("\n=== PROMPT SENT TO MODEL ===\n")
    for m in messages:
        print(f"[{m['role'].upper()}]")
        print(m["content"])
        print()

    output = generate_chat(model, tokenizer, messages)

    print("\n=== MODEL OUTPUT ===\n")
    print(output)

if __name__ == "__main__":
    main()
