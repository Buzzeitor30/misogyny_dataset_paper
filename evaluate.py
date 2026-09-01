import argparse
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 4095


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-only evaluation of an LLM on the misogyny detection dataset."
    )
    parser.add_argument(
        "--test_file",
        default="data/test.csv",
        help="Path to the evaluation CSV file (default: data/test.csv)",
    )
    parser.add_argument(
        "--prompt_file",
        default="prompts/baseline.md",
        help="Path to the prompt template file (default: prompts/baseline.md)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model identifier or local path to run inference with",
    )
    return parser.parse_args()


def parse_response(raw_response):
    reasoning_match = re.search(
        r"Reasoning:\s*(.*?)(?=\n\s*Misogyny:|\Z)", raw_response, re.DOTALL
    )
    prediction_match = re.search(r"Misogyny:\s*(NM|M)\b", raw_response)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    prediction = prediction_match.group(1) if prediction_match else ""
    return prediction, reasoning


def main():
    args = parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    df = pd.read_csv(args.test_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="fp16",
        device_map="auto",
    )
    model.eval()

    do_sample = args.temperature > 0

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=args.model_name):
        prompt = prompt_template.format(
            song_lyrics_title=row["song_title"], lyrics=row["lyrics"]
        )
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        generate_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        prediction, reasoning = parse_response(raw_response)

        results.append(
            {
                "raw_response": raw_response,
                "song_id": row["song_id"],
                "lyrics": row["lyrics"],
                "is_misogynistic": row["is_misogynistic"],
                "song_title": row["song_title"],
                "is_misogynistic_pred": prediction,
                "is_misogynistic_reasoning": reasoning,
            }
        )

    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    safe_model_name = args.model_name.replace("/", "_")
    output_path = os.path.join(
        predictions_dir, f"{safe_model_name}_{args.temperature}.csv"
    )

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
