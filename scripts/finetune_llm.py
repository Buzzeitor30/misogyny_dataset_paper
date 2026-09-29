import argparse
import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

INVALID = "INVALID"
NONE = "NONE"
IM_END = "<|im_end|>"

MAX_NEW_TOKENS = 48
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
LORA_DROPOUT = 0.05
# Language-model projections only (attention, Gated-DeltaNet, MLP); the vision tower
# is left untouched. in_proj_a / in_proj_b are skipped: their output is one value per head.
LORA_TARGET_REGEX = (
    r".*language_model.*\."
    r"(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"
)

# ----------------------------------------------------------------------------------
# Prompts
#
# Task mapping onto data/train.csv columns:
#   task1 -> is_misogynistic                                     (M / NM)
#   task2 -> type_sexualization, type_violence, type_hate        (multi-label, M songs only)
#   task3 -> has_gender_stereotype                               (Y / N, M songs only)
#   all   -> the three tasks at once; task2/task3 fields are NONE for NM songs
# ----------------------------------------------------------------------------------

ROLE = """# ROLE
You are an expert at detecting misogynistic content in Spanish song lyrics."""

MISOGYNY_DEFINITION = """Misogyny is defined as any manifestation that expresses contempt, hostility, or disparagement towards women on the basis of their gender. It can be observed in the form of insults, mockery, objectification, violence, or judgements that reinforce their subordination in various social and cultural contexts."""

TASK1_DESCRIPTION = """Classify the lyrics into one of two categories:
- NM (No-Misogynistic): The lyrics do NOT contain content that expresses contempt, hostility, disparagement, objectification, or subordination of women based on their gender.
- M (Misogynistic): The lyrics DO contain content that expresses contempt, hostility, disparagement, objectification, or subordination of women based on their gender."""

TASK2_DESCRIPTION = """The lyrics are misogynistic. Identify which types of misogyny they contain. The types are not mutually exclusive: a song can contain one, two, or all three of them.
- Sexualization: The lyrics reduce women to sexual objects or to their bodies, or treat them as available for sexual consumption.
- Violence: The lyrics express, describe, incite, or normalize physical, psychological, or sexual violence against women.
- Hate: The lyrics express hatred, contempt, insults, or hostility towards women, as a group or individually, because of their gender."""

TASK3_DESCRIPTION = """The lyrics are misogynistic. Determine whether they reinforce gender stereotypes.
- Y (Yes): The lyrics present women through rigid or generalized roles, traits, or behaviours attributed to their gender (for example: submissive, emotional, deceitful, defined by their looks, confined to domestic or caring roles, or as property of men).
- N (No): The lyrics do NOT reinforce gender stereotypes."""

ALL_DESCRIPTION = """Complete the following three tasks.

Task 1 - Misogyny. Classify the lyrics into one of two categories:
- NM (No-Misogynistic): The lyrics do NOT contain content that expresses contempt, hostility, disparagement, objectification, or subordination of women based on their gender.
- M (Misogynistic): The lyrics DO contain content that expresses contempt, hostility, disparagement, objectification, or subordination of women based on their gender.

Task 2 - Types of misogyny. Only for misogynistic lyrics. The types are not mutually exclusive: a song can contain one, two, or all three of them.
- Sexualization: The lyrics reduce women to sexual objects or to their bodies, or treat them as available for sexual consumption.
- Violence: The lyrics express, describe, incite, or normalize physical, psychological, or sexual violence against women.
- Hate: The lyrics express hatred, contempt, insults, or hostility towards women, as a group or individually, because of their gender.

Task 3 - Gender stereotypes. Only for misogynistic lyrics.
- Y (Yes): The lyrics present women through rigid or generalized roles, traits, or behaviours attributed to their gender (for example: submissive, emotional, deceitful, defined by their looks, confined to domestic or caring roles, or as property of men).
- N (No): The lyrics do NOT reinforce gender stereotypes.

If the lyrics are NM, the answer of Task 2 and Task 3 must be NONE."""

TASK1_FORMAT = "Misogyny: <NM or M>"
TASK2_FORMAT = """Sexualization: <Yes or No>
Violence: <Yes or No>
Hate: <Yes or No>"""
TASK3_FORMAT = "Stereotype: <Y or N>"
ALL_FORMAT = """Misogyny: <NM or M>
Sexualization: <Yes, No or NONE>
Violence: <Yes, No or NONE>
Hate: <Yes, No or NONE>
Stereotype: <Y, N or NONE>"""

PROMPT_TEMPLATE = """{role}

# TASK DESCRIPTION
{definition}

{description}

# OUTPUT FORMAT
Respond strictly in the following format, and nothing else:

{output_format}

# INPUT
<song_title>
{{title}}
</song_title>

<song_lyrics>
{{lyrics}}
</song_lyrics>"""

PROMPTS = {
    task: PROMPT_TEMPLATE.format(
        role=ROLE,
        definition=MISOGYNY_DEFINITION,
        description=description,
        output_format=output_format,
    )
    for task, description, output_format in [
        ("task1", TASK1_DESCRIPTION, TASK1_FORMAT),
        ("task2", TASK2_DESCRIPTION, TASK2_FORMAT),
        ("task3", TASK3_DESCRIPTION, TASK3_FORMAT),
        ("all", ALL_DESCRIPTION, ALL_FORMAT),
    ]
}

# (answer key, dataframe column, valid answers)
MISOGYNY_FIELD = ("Misogyny", "is_misogynistic", ("M", "NM"))
TYPE_FIELDS = [
    ("Sexualization", "type_sexualization", ("Yes", "No")),
    ("Violence", "type_violence", ("Yes", "No")),
    ("Hate", "type_hate", ("Yes", "No")),
]
STEREOTYPE_FIELD = ("Stereotype", "has_gender_stereotype", ("Y", "N"))

FIELDS = {
    "task1": [MISOGYNY_FIELD],
    "task2": TYPE_FIELDS,
    "task3": [STEREOTYPE_FIELD],
    "all": [MISOGYNY_FIELD] + TYPE_FIELDS + [STEREOTYPE_FIELD],
}
# In "all" mode the task2/task3 answers may also be NONE (gold for NM songs).
VALID_ANSWERS = {
    "task1": {"Misogyny": ("M", "NM")},
    "task2": {name: values for name, _, values in TYPE_FIELDS},
    "task3": {"Stereotype": ("Y", "N")},
    "all": {
        "Misogyny": ("M", "NM"),
        **{name: values + (NONE,) for name, _, values in TYPE_FIELDS},
        "Stereotype": ("Y", "N", NONE),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA-finetune Qwen3.5 on the misogyny dataset (task1, task2, task3 or all three at once)."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["task1", "task2", "task3", "all"],
        help="task1: misogyny presence; task2: misogyny types; task3: gender stereotype; "
        "all: the three tasks at once (task2/task3 answers are NONE for NM songs)",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--train_file", default="data/train.csv")
    parser.add_argument("--test_file", default="data/test.csv")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank r; alpha is always 2*r")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=4, help="Maximum number of epochs")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Stop after this many epochs without validation F1 improvement",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4, help="Songs per forward+backward pass")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Batches accumulated per optimizer step (effective batch size = batch_size * this)",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Attention implementation passed to from_pretrained (default: flash_attention_2)",
    )
    parser.add_argument("--output_dir", default="models/finetuned")
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument(
        "--max_train_samples", type=int, default=None, help="Debug: cap the number of training songs"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Debug: cap the number of validation and test songs",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------


def load_songs(path):
    """Load a split and add one `gold_<Answer key>` column per answer field.

    Types and stereotype are only annotated for M songs; NM songs get NONE.
    """
    df = pd.read_csv(path)
    df["song_title"] = df["song_title"].fillna("")
    df["lyrics"] = df["lyrics"].fillna("")

    is_m = df["is_misogynistic"] == "M"
    df["gold_Misogyny"] = df["is_misogynistic"]
    for name, column, _ in TYPE_FIELDS:
        gold = df[column].map({1.0: "Yes", 0.0: "No"})
        df[f"gold_{name}"] = gold.where(is_m, NONE)
    df["gold_Stereotype"] = df["has_gender_stereotype"].where(is_m, NONE)

    unlabeled = df.loc[is_m, [f"gold_{n}" for n, _, _ in TYPE_FIELDS + [STEREOTYPE_FIELD]]].isna().any(axis=1)
    if unlabeled.any():
        raise ValueError(f"{path}: {int(unlabeled.sum())} M songs are missing type/stereotype labels")
    return df


def make_split(df, val_fraction, seed):
    """Stratified train/validation split over task1 x task2 x task3 x country of origin.

    The key is task1 | types | stereotype | language (es_ES = Spain, es_LATAM = Latin America).
    Strata that are too small to reach the validation set fall back to a coarser key.
    """
    min_count = math.ceil(1 / val_fraction)
    types = df[[f"gold_{n}" for n, _, _ in TYPE_FIELDS]].apply(
        lambda r: "".join("1" if v == "Yes" else "0" if v == "No" else "-" for v in r), axis=1
    )
    fine = df["gold_Misogyny"] + "|" + types + "|" + df["gold_Stereotype"] + "|" + df["language"]
    medium = df["gold_Misogyny"] + "|" + df["gold_Stereotype"] + "|" + df["language"]
    coarse = df["gold_Misogyny"] + "|" + df["language"]

    key = fine.where(fine.map(fine.value_counts()) >= min_count, medium)
    key = key.where(key.map(key.value_counts()) >= min_count, coarse)

    train_idx, val_idx = train_test_split(
        df.index, test_size=val_fraction, stratify=key, random_state=seed
    )
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)


def report_split(train_df, val_df):
    for name in ["Misogyny", "Stereotype", "Sexualization", "Violence", "Hate"]:
        col = f"gold_{name}"
        table = pd.concat(
            {
                "train": train_df[col].value_counts(normalize=True),
                "val": val_df[col].value_counts(normalize=True),
            },
            axis=1,
        )
        print(f"Split proportions ({name}):\n{(table * 100).round(1).to_string()}\n", flush=True)
    table = pd.concat(
        {
            "train": train_df["language"].value_counts(normalize=True),
            "val": val_df["language"].value_counts(normalize=True),
        },
        axis=1,
    )
    print(f"Split proportions (language):\n{(table * 100).round(1).to_string()}\n", flush=True)


def build_target(row, task):
    return "\n".join(f"{name}: {row[f'gold_{name}']}" for name, _, _ in FIELDS[task])


def encode_example(tokenizer, task, title, lyrics, max_seq_length, target_ids):
    """Tokenize the chat prompt (thinking disabled); lyrics are truncated if the example is too long."""

    def encode_prompt(lyrics_text):
        messages = [
            {"role": "user", "content": PROMPTS[task].format(title=title, lyrics=lyrics_text)}
        ]
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prompt_ids = encode_prompt(lyrics)
    overflow = len(prompt_ids) + len(target_ids) - max_seq_length
    if overflow > 0:
        lyrics_ids = tokenizer(lyrics, add_special_tokens=False)["input_ids"]
        keep = max(len(lyrics_ids) - overflow - 16, 0)
        while True:
            prompt_ids = encode_prompt(tokenizer.decode(lyrics_ids[:keep]))
            if len(prompt_ids) + len(target_ids) <= max_seq_length or keep == 0:
                break
            keep = max(keep - 64, 0)
    return prompt_ids


def encode_dataframe(tokenizer, df, task, max_seq_length):
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    examples = []
    for _, row in df.iterrows():
        target = build_target(row, task)
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [im_end_id]
        prompt_ids = encode_example(
            tokenizer, task, row["song_title"], row["lyrics"], max_seq_length, target_ids
        )
        examples.append({"prompt_ids": prompt_ids, "target_ids": target_ids})
    return examples


def collate_train(examples, pad_token_id):
    """Left-pad so every answer sits in the last positions of its row."""
    length = max(len(e["prompt_ids"]) + len(e["target_ids"]) for e in examples)
    input_ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
    labels = torch.full((len(examples), length), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        ids = e["prompt_ids"] + e["target_ids"]
        input_ids[i, length - len(ids) :] = torch.tensor(ids)
        attention_mask[i, length - len(ids) :] = 1
        labels[i, length - len(e["target_ids"]) :] = torch.tensor(e["target_ids"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_prompts(examples, pad_token_id):
    length = max(len(e["prompt_ids"]) for e in examples)
    input_ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
    for i, e in enumerate(examples):
        ids = e["prompt_ids"]
        input_ids[i, length - len(ids) :] = torch.tensor(ids)
        attention_mask[i, length - len(ids) :] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ----------------------------------------------------------------------------------
# Loss, batch size probe and training
# ----------------------------------------------------------------------------------


def answer_loss_sum(model, batch):
    """Summed cross-entropy over the answer tokens.

    The vocabulary is huge (248K), so logits are only computed for the last k+1 positions,
    where k is the longest answer in the batch. This is valid because of left padding.
    """
    labels = batch["labels"]
    k = int((labels != -100).sum(dim=1).max())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            logits_to_keep=k + 1,
            use_cache=False,
        )
    logits = out.logits[:, :-1].float()
    targets = labels[:, -k:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, int((targets != -100).sum())


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def forward_backward(model, examples, micro_batch_size, pad_token_id, denominator, device):
    """Accumulate the gradients of one optimizer batch over micro-batches, returning the summed loss.

    The batch is sorted by length so each micro-batch pads to similar lengths. `denominator` is
    the answer-token count of the whole batch, so the gradient does not depend on the micro-batch
    size. On OOM the whole step is redone with micro-batches of half the size, so a rare very long
    micro-batch cannot kill the run.
    """
    examples = sorted(examples, key=lambda e: len(e["prompt_ids"]) + len(e["target_ids"]), reverse=True)
    chunk = micro_batch_size
    while True:
        try:
            total = 0.0
            for start in range(0, len(examples), chunk):
                batch = to_device(collate_train(examples[start : start + chunk], pad_token_id), device)
                loss, _ = answer_loss_sum(model, batch)
                (loss / denominator).backward()
                total += loss.item()
            return total
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if chunk == 1:
                raise
            chunk = max(chunk // 2, 1)
            print(f"OOM during a training step, retrying with micro-batches of {chunk}", flush=True)


def random_batches(n_examples, batch_size, rng):
    """Shuffle and cut into optimizer batches.

    Batches are not grouped by length: song length correlates with the label (M songs are
    longer), so length-grouped batches would be label-skewed. Padding is reduced inside each
    batch instead, when it is split into micro-batches.
    """
    indices = list(range(n_examples))
    rng.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, n_examples, batch_size)]


def train_one_epoch(
    model, examples, effective_batch_size, micro_batch_size, optimizer, scheduler, pad_token_id, device, rng, epoch
):
    model.train()
    batches = random_batches(len(examples), effective_batch_size, rng)
    total_loss, total_tokens = 0.0, 0
    progress = tqdm(batches, desc=f"epoch {epoch}")
    for batch_indices in progress:
        batch_examples = [examples[i] for i in batch_indices]
        n_tokens = sum(len(e["target_ids"]) for e in batch_examples)
        loss_sum = forward_backward(
            model, batch_examples, micro_batch_size, pad_token_id, n_tokens, device
        )
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss_sum
        total_tokens += n_tokens
        progress.set_postfix(loss=f"{loss_sum / n_tokens:.4f}")
    return total_loss / total_tokens


# ----------------------------------------------------------------------------------
# Generation and metrics
# ----------------------------------------------------------------------------------


def parse_response(raw_response, task):
    """Extract one answer per field; anything missing or outside the valid answers is INVALID."""
    parsed = {}
    for name, _, _ in FIELDS[task]:
        match = re.search(rf"{name}:\s*([A-Za-z]+)", raw_response)
        answer = match.group(1) if match else ""
        parsed[name] = answer if answer in VALID_ANSWERS[task][name] else INVALID
    return parsed


@torch.no_grad()
def generate_batch(model, tokenizer, examples, device):
    pad_token_id = tokenizer.pad_token_id
    batch = to_device(collate_prompts(examples, pad_token_id), device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.convert_tokens_to_ids(IM_END),
            pad_token_id=pad_token_id,
            use_cache=True,
        )
    generated = output_ids[:, batch["input_ids"].shape[1] :]
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]


def generate_responses(model, tokenizer, examples, batch_size, device):
    """Greedy answers for all examples (longest first); on OOM the batch is halved."""
    model.eval()
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["prompt_ids"]), reverse=True)
    responses = [None] * len(examples)
    for start in tqdm(range(0, len(order), batch_size), desc="generate"):
        batch_indices = order[start : start + batch_size]
        chunk = len(batch_indices)
        while True:
            try:
                out = []
                for i in range(0, len(batch_indices), chunk):
                    out += generate_batch(
                        model, tokenizer, [examples[j] for j in batch_indices[i : i + chunk]], device
                    )
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk == 1:
                    raise
                chunk = max(chunk // 2, 1)
        for i, response in zip(batch_indices, out):
            responses[i] = response
    return responses


def macro_f1(gold, pred, labels):
    return f1_score(gold, pred, labels=labels, average="macro", zero_division=0)


def positive_f1(gold, pred, positive):
    return f1_score(gold, pred, labels=[positive], average="macro", zero_division=0)


def compute_metrics(gold_df, pred_df, task):
    """Validation/test scores. `f1` is the early-stopping metric.

    task1: macro-F1 over {M, NM}
    task2: mean over the three types of the F1 of the positive (Yes) class
    task3: macro-F1 over {Y, N}
    all:   mean of the three scores above; task2/task3 are scored on gold-M songs only
    """
    metrics = {}
    # task2/task3 are only defined for M songs (in "all" mode gold_Misogyny tells which)
    is_m = (
        (gold_df["gold_Misogyny"] == "M").to_numpy()
        if task == "all"
        else np.ones(len(gold_df), dtype=bool)
    )

    def gold_of(name, mask=None):
        col = gold_df[f"gold_{name}"]
        return (col if mask is None else col[mask]).tolist()

    def pred_of(name, mask=None):
        col = pred_df[name]
        return (col if mask is None else col[mask]).tolist()

    scores = {}
    if task in ("task1", "all"):
        scores["task1"] = macro_f1(gold_of("Misogyny"), pred_of("Misogyny"), ["M", "NM"])
    if task in ("task2", "all"):
        type_f1 = {}
        for name, _, _ in TYPE_FIELDS:
            type_f1[name] = positive_f1(gold_of(name, is_m), pred_of(name, is_m), "Yes")
            metrics[f"f1_{name.lower()}"] = type_f1[name]
        scores["task2"] = float(np.mean(list(type_f1.values())))
    if task in ("task3", "all"):
        scores["task3"] = macro_f1(gold_of("Stereotype", is_m), pred_of("Stereotype", is_m), ["Y", "N"])

    for name, value in scores.items():
        metrics[f"f1_{name}"] = value
    metrics["f1"] = float(np.mean(list(scores.values())))
    metrics["invalid_rate"] = float(
        (pred_df[[name for name, _, _ in FIELDS[task]]] == INVALID).any(axis=1).mean()
    )
    return metrics


def evaluate(model, tokenizer, examples, gold_df, task, batch_size, device):
    responses = generate_responses(model, tokenizer, examples, batch_size, device)
    pred_df = pd.DataFrame([parse_response(r, task) for r in responses])
    return compute_metrics(gold_df, pred_df, task), responses, pred_df


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------


def check_attention_backend(attn_implementation):
    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "attn_implementation=flash_attention_2 requires the flash-attn package "
                "(install the wheel on the cluster), or pass --attn_implementation sdpa explicitly."
            ) from exc


def main():
    args = parse_args()
    task = args.task
    check_attention_backend(args.attn_implementation)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")

    safe_model_name = args.model_name.replace("/", "_")
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    run_name = f"{safe_model_name}_{task}_r{args.lora_rank}_lr{args.learning_rate:g}_bs{effective_batch_size}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.predictions_dir, exist_ok=True)
    print(f"Run: {run_name}", flush=True)

    # Data. The split is made on the full train set (so validation songs are identical for
    # every task); task2 and task3 are then restricted to M songs.
    full_train = load_songs(args.train_file)
    train_df, val_df = make_split(full_train, args.val_fraction, args.seed)
    report_split(train_df, val_df)
    pd.DataFrame({"song_id": val_df["song_id"]}).to_csv(
        os.path.join(run_dir, "val_song_ids.csv"), index=False
    )
    test_df = load_songs(args.test_file)

    if task in ("task2", "task3"):
        train_df, val_df, test_df = (
            d[d["is_misogynistic"] == "M"].reset_index(drop=True) for d in (train_df, val_df, test_df)
        )
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=args.seed)
    if args.max_eval_samples:
        val_df = val_df.head(args.max_eval_samples)
        test_df = test_df.head(args.max_eval_samples)
    train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))
    print(f"Songs: train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_examples = encode_dataframe(tokenizer, train_df, task, args.max_seq_length)
    val_examples = encode_dataframe(tokenizer, val_df, task, args.max_seq_length)
    test_examples = encode_dataframe(tokenizer, test_df, task, args.max_seq_length)
    max_len = max(len(e["prompt_ids"]) + len(e["target_ids"]) for e in train_examples)
    print(f"Longest training example: {max_len} tokens", flush=True)

    # Model
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": 0},
    )
    print(f"attn_implementation={model.config._attn_implementation}", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_REGEX,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Batch size {args.batch_size} x {args.gradient_accumulation_steps} accumulation steps "
        f"= effective batch size {effective_batch_size}",
        flush=True,
    )

    steps_per_epoch = math.ceil(len(train_examples) / effective_batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(WARMUP_RATIO * steps_per_epoch * args.num_epochs),
        num_training_steps=steps_per_epoch * args.num_epochs,
    )

    # Training with early stopping on the validation F1 (evaluated after every epoch)
    history = []
    best_f1, best_epoch, best_state, epochs_without_improvement = -1.0, 0, None, 0
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_examples, effective_batch_size, args.batch_size, optimizer, scheduler,
            tokenizer.pad_token_id, device, rng, epoch,
        )
        val_metrics, _, _ = evaluate(
            model, tokenizer, val_examples, val_df, task, args.eval_batch_size, device
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val={json.dumps(val_metrics)}", flush=True)

        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch, epochs_without_improvement = val_metrics["f1"], epoch, 0
            best_state = {
                n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping after epoch {epoch} (best epoch: {best_epoch})", flush=True)
                break

    # Restore the best epoch, save the adapter and evaluate on the test set
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best_state[n])
    model.save_pretrained(os.path.join(run_dir, "adapter"))

    test_metrics, responses, pred_df = evaluate(
        model, tokenizer, test_examples, test_df, task, args.eval_batch_size, device
    )
    print(f"Test (best epoch {best_epoch}): {json.dumps(test_metrics)}", flush=True)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "effective_batch_size": effective_batch_size,
                "best_epoch": best_epoch,
                "best_val_f1": best_f1,
                "history": history,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    predictions = pd.DataFrame(
        {
            "raw_response": responses,
            "song_id": test_df["song_id"],
            "song_title": test_df["song_title"],
        }
    )
    for name, column, _ in FIELDS[task]:
        predictions[column] = test_df[f"gold_{name}"]
        predictions[f"{column}_pred"] = pred_df[name]
    predictions_path = os.path.join(args.predictions_dir, f"finetuned_{run_name}.csv")
    predictions.to_csv(predictions_path, index=False)
    print(f"Saved {len(predictions)} predictions to {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
