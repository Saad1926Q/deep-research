import argparse
import pandas as pd
from dspy import Example
from tqdm import tqdm

from evals.metrics import rubric_metric

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    required=True,
    help="name of the run to evaluate (e.g. 'baseline'). expects reports at reports/<name>/<sample_id>.md and saves results to evals/<name>_results.csv",
)
args = parser.parse_args()

df = pd.read_json(
    "hf://datasets/ScaleAI/researchrubrics/processed_data.jsonl", lines=True
).sample(5, random_state=42)

scores = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating reports"):
    idx = row["sample_id"]
    report_file = f"reports/{args.name}/{idx}.md"
    with open(report_file, "r") as f:
        report = f.read()

    example = Example(
        rubrics=row["rubrics"],
        report=report,
        research_request=row["prompt"],
    )

    score = rubric_metric(example, prediction=None)
    scores.append({"sample_id": idx, "score": score})
    print(f"{idx}: {score:.3f}")

results_df = pd.DataFrame(scores)
results_df.to_csv(f"evals/{args.name}_results.csv", index=False)
