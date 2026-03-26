import argparse
import asyncio
import pickle

from tqdm import tqdm

from main import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="run name e.g. baseline, v2")
args = parser.parse_args()

with open("evals/clarifications.pkl", "rb") as f:
    clarifications = pickle.load(f)


async def generate():
    for sample_id, data in tqdm(clarifications.items(), desc="Generating reports"):
        await run_pipeline(
            eval=True,
            research_request=data["research_request"],
            name=args.name,
            idx=sample_id,
            clarifying_questions_and_answers=data["clarifying_questions_and_answers"],
        )


asyncio.run(generate())
