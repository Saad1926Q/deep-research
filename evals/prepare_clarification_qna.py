import pickle

import dspy
import pandas as pd

from src.config import pipeline_lm
from src.modules import clarifier
from src.utils import answer_question

dspy.configure(lm=pipeline_lm)

df = pd.read_json(
    "hf://datasets/ScaleAI/researchrubrics/processed_data.jsonl", lines=True
).sample(5, random_state=42)

clarifications = {}

for _, row in df.iterrows():
    sample_id = row["sample_id"]
    research_request = row["prompt"]
    print(f"Generating clarifications for {sample_id}...")

    clarifier_result = clarifier(research_request=research_request)
    clarifying_questions_and_answers = [
        {
            "question": q,
            "answer": answer_question(
                research_request=research_request, question=q
            ).answer,
        }
        for q in clarifier_result.clarifying_questions
    ]
    clarifications[sample_id] = {
        "research_request": research_request,
        "clarifying_questions_and_answers": clarifying_questions_and_answers,
    }
    print(f"  done — {len(clarifying_questions_and_answers)} questions")

with open("evals/clarifications.pkl", "wb") as f:
    pickle.dump(clarifications, f)
