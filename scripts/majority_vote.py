import argparse
import glob
import os
import re

import pandas as pd

POSITIVE = "M"
NEGATIVE = "NM"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Majority-vote ensemble over per-temperature prediction files."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model identifier used at inference time (e.g. Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--predictions_dir",
        default="predictions",
        help="Directory holding the per-run prediction CSVs (default: predictions)",
    )
    parser.add_argument(
        "--temperatures",
        nargs="*",
        type=float,
        default=None,
        help="Subset of temperatures to vote over (default: every run found on disk)",
    )
    parser.add_argument(
        "--min_votes",
        type=int,
        default=None,
        help="Votes for M needed to predict M (default: strict majority, floor(n/2)+1)",
    )
    return parser.parse_args()


def collect_runs(predictions_dir, model_name, temperatures):
    safe_model_name = model_name.replace("/", "_")
    pattern = os.path.join(predictions_dir, f"{safe_model_name}_*.csv")

    runs = {}
    for path in sorted(glob.glob(pattern)):
        match = re.fullmatch(rf"{re.escape(safe_model_name)}_(.+)\.csv", os.path.basename(path))
        if not match:
            continue
        try:
            temperature = float(match.group(1))
        except ValueError:
            continue  # skip ensemble outputs written by earlier runs
        if temperatures is not None and temperature not in temperatures:
            continue
        runs[temperature] = pd.read_csv(path)

    if temperatures is not None:
        missing = sorted(set(temperatures) - set(runs))
        if missing:
            raise SystemExit(f"No prediction file found for temperatures: {missing}")
    if not runs:
        raise SystemExit(f"No prediction files matched {pattern}")

    return dict(sorted(runs.items()))


def build_vote_table(runs):
    temperatures = list(runs)
    reference = runs[temperatures[0]]

    votes = reference[["song_id", "song_title", "is_misogynistic"]].copy()
    for temperature, df in runs.items():
        column = df[["song_id", "is_misogynistic_pred"]].rename(
            columns={"is_misogynistic_pred": f"pred_{temperature}"}
        )
        votes = votes.merge(column, on="song_id", how="left", validate="one_to_one")

    vote_columns = [f"pred_{t}" for t in temperatures]
    votes["votes_M"] = (votes[vote_columns] == POSITIVE).sum(axis=1)
    votes["votes_cast"] = votes[vote_columns].isin([POSITIVE, NEGATIVE]).sum(axis=1)
    return votes, vote_columns


def score(y_true, y_pred):
    def per_class(label):
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    precision_m, recall_m, f1_m = per_class(POSITIVE)
    precision_nm, recall_nm, f1_nm = per_class(NEGATIVE)

    return {
        "acc": (y_true == y_pred).mean(),
        "P_M": precision_m,
        "R_M": recall_m,
        "F1_M": f1_m,
        "F1_NM": f1_nm,
        "macroF1": (f1_m + f1_nm) / 2,
        "TP": int(((y_true == POSITIVE) & (y_pred == POSITIVE)).sum()),
        "FP": int(((y_true == NEGATIVE) & (y_pred == POSITIVE)).sum()),
        "FN": int(((y_true == POSITIVE) & (y_pred == NEGATIVE)).sum()),
        "TN": int(((y_true == NEGATIVE) & (y_pred == NEGATIVE)).sum()),
    }


def main():
    args = parse_args()

    runs = collect_runs(args.predictions_dir, args.model_name, args.temperatures)
    votes, vote_columns = build_vote_table(runs)
    n_runs = len(runs)

    print(f"Voting over {n_runs} runs at temperatures {list(runs)} ({len(votes)} songs)\n")

    incomplete = int((votes["votes_cast"] < n_runs).sum())
    if incomplete:
        print(f"Warning: {incomplete} songs have an unparsed label in at least one run\n")

    print("Single runs:")
    rows = []
    for temperature, column in zip(runs, vote_columns):
        rows.append({"scheme": f"temp {temperature}", **score(votes["is_misogynistic"], votes[column])})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nVote thresholds (predict M when votes_M >= k):")
    rows = []
    for k in range(1, n_runs + 1):
        predicted = (votes["votes_M"] >= k).map({True: POSITIVE, False: NEGATIVE})
        label = f"k={k}" + (" (majority)" if k == n_runs // 2 + 1 else "")
        rows.append({"scheme": label, **score(votes["is_misogynistic"], predicted)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    min_votes = args.min_votes if args.min_votes is not None else n_runs // 2 + 1
    votes["is_misogynistic_pred"] = (votes["votes_M"] >= min_votes).map(
        {True: POSITIVE, False: NEGATIVE}
    )
    votes["unanimous"] = votes["votes_M"].isin([0, n_runs])

    safe_model_name = args.model_name.replace("/", "_")
    output_path = os.path.join(
        args.predictions_dir, f"{safe_model_name}_vote{n_runs}_k{min_votes}.csv"
    )
    votes.to_csv(output_path, index=False)

    agreement = votes["unanimous"].mean()
    print(f"\nUnanimous across all {n_runs} runs: {agreement:.1%} of songs")
    print(f"Saved ensemble (k={min_votes}) to {output_path}")


if __name__ == "__main__":
    main()
