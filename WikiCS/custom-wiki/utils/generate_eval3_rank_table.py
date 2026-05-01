import csv
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(r"C:\programming\github-repos\graph-ending")
EVAL3_DIR = REPO_ROOT / "WikiCS" / "custom-wiki" / "cache" / "eval-3"
INPUT_CSV = EVAL3_DIR / "combined_metrics.csv"
OUTPUT_CSV = EVAL3_DIR / "model_rankings.csv"
OUTPUT_TEX = EVAL3_DIR / "model_rankings_rows.tex"


HIGHER_IS_BETTER = [
    "Hit@1",
    "Hit@10",
    "Hit@50",
    "Hit@100",
    "MRR",
    "NDCG",
    "Recall",
    "MAP",
    "Aggregate Diversity",
    "Novelty",
    "Intra-List Distance",
]

LOWER_IS_BETTER = [
    "Popularity Lift",
    "Gini Index",
]

METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    rank_df = pd.DataFrame()
    rank_df["Model"] = df["Model"]

    for metric in HIGHER_IS_BETTER:
        rank_df[metric] = df[metric].rank(method="min", ascending=False).astype(int)

    for metric in LOWER_IS_BETTER:
        rank_df[metric] = df[metric].rank(method="min", ascending=True).astype(int)

    rank_df["Total Score"] = rank_df[METRICS].sum(axis=1)
    rank_df["Total Rank"] = rank_df["Total Score"].rank(method="min", ascending=True).astype(int)
    rank_df = rank_df.sort_values(["Total Score", "Model"], ascending=[True, True]).reset_index(drop=True)

    ordered_cols = ["Model", "Total Rank", "Total Score"] + METRICS
    rank_df.to_csv(OUTPUT_CSV, index=False, columns=ordered_cols, quoting=csv.QUOTE_MINIMAL)

    row_cols = ["Total Rank"] + METRICS
    tex_lines = []
    for _, row in rank_df.iterrows():
        values = [latex_escape(str(row["Model"]))] + [str(int(row[col])) for col in row_cols]
        tex_lines.append(" & ".join(values) + r" \\")

    OUTPUT_TEX.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
