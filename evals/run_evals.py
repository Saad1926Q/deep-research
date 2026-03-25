import asyncio
import json

import dspy
import pandas as pd
from tqdm import tqdm

from evals.metrics import (
    annotator_metric,
    clarifier_metric,
    planner_metric,
    processor_relevance_metric,
    synthesizer_metric,
)
from datasets.splits import TEST
from src.config import pipeline_lm
from src.modules import annotator, clarifier, planner, processor, synthesizer

dspy.configure(lm=pipeline_lm)


def run_clarifier_evals(snapshots):
    print("Running Evals for Clarifier")
    results = []
    for snap in tqdm(snapshots):
        research_request = snap["research_request"]

        prediction = clarifier(research_request=research_request)
        example = dspy.Example(research_request=research_request).with_inputs(
            "research_request"
        )
        score = clarifier_metric(example, prediction)

        results.append(
            {
                "research_request": research_request,
                "clarifier_score": score,
                "clarifying_questions": prediction.clarifying_questions,
            }
        )
        tqdm.write(f"  score: {score:.3f}")

    avg = sum(r["clarifier_score"] for r in results) / len(results)
    print(f"Clarifier Avg Score : {avg:.3f}\n")
    return results


def run_planner_evals(snapshots):
    print("Running Evals for Planner")
    results = []
    for snap in tqdm(snapshots):
        research_request = snap["research_request"]
        clarifying_questions_and_answers = snap["clarifying_questions_and_answers"]

        prediction = planner(
            research_request=research_request,
            clarifying_questions_and_answers=clarifying_questions_and_answers,
            max_num_research_topics=5,
        )
        example = dspy.Example(research_request=research_request).with_inputs(
            "research_request"
        )
        score = planner_metric(example, prediction)

        results.append(
            {
                "research_request": research_request,
                "planner_score": score,
                "research_topics": prediction.research_topics,
            }
        )
        tqdm.write(f"  score: {score:.3f}")

    avg = sum(r["planner_score"] for r in results) / len(results)
    print(f"Planner Avg Score : {avg:.3f}\n")
    return results


async def _process_source(research_request: str, topic: str, source: dict) -> dict:
    url = source["url"]
    page_content = source["page_content"]
    is_hard_negative = source.get("is_hard_negative", False)

    prediction = await processor.aforward(
        research_task=research_request,
        research_subtask=topic,
        url=url,
        page_content=page_content,
    )
    example = dspy.Example(
        research_task=research_request,
        research_subtask=topic,
        page_content=page_content,
        is_hard_negative=is_hard_negative,
    ).with_inputs(
        "research_task", "research_subtask", "page_content", "is_hard_negative"
    )

    relevance_score = processor_relevance_metric(example, prediction)

    return {
        "topic": topic,
        "url": url,
        "is_hard_negative": is_hard_negative,
        "page_is_relevant": prediction.page_is_relevant,
        "summary": prediction.summary,
        "relevant_facts": prediction.relevant_facts,
        "relevance_score": relevance_score,
    }


async def run_processor_evals(snapshots):
    print("Running Evals for Processor")
    results = []
    for snap in tqdm(snapshots):
        research_request = snap["research_request"]

        source_results = []
        for topic_entry in snap["topic_sources"]:
            for source in topic_entry["sources"]:
                result = await _process_source(research_request, topic_entry["topic"], source)
                source_results.append(result)

        avg_relevance = sum(s["relevance_score"] for s in source_results) / len(
            source_results
        )

        results.append(
            {
                "research_request": research_request,
                "processor_relevance_score": avg_relevance,
                "processor_outputs": list(source_results),
            }
        )
        tqdm.write(f"  relevance: {avg_relevance:.3f}")

    avg_r = sum(r["processor_relevance_score"] for r in results) / len(results)
    print(f"Processor Avg Relevance    : {avg_r:.3f}\n")
    return results


def run_synthesizer_evals(snapshots, processor_results):
    print("Running Evals for Synthesizer")
    results = []
    for snap, proc in tqdm(zip(snapshots, processor_results), total=len(snapshots)):
        research_request = snap["research_request"]
        clarifying_questions_and_answers = snap["clarifying_questions_and_answers"]
        research_topics = [t["topic"] for t in snap["topic_sources"]]

        topics = {}
        for s in proc["processor_outputs"]:
            if not s["page_is_relevant"]:
                continue
            if s["topic"] not in topics:
                topics[s["topic"]] = []
            topics[s["topic"]].append(
                {
                    "url": s["url"],
                    "summary": s["summary"],
                    "relevant_facts": s["relevant_facts"],
                }
            )
        gathered_findings = [
            {"subtopic": topic, "sources": sources} for topic, sources in topics.items()
        ]

        prediction = synthesizer(
            research_request=research_request,
            clarifying_questions_and_answers=clarifying_questions_and_answers,
            research_topics=research_topics,
            gathered_findings=gathered_findings,
        )
        example = dspy.Example(
            research_request=research_request,
            research_topics=research_topics,
            gathered_findings=gathered_findings,
        ).with_inputs("research_request", "research_topics", "gathered_findings")
        score = synthesizer_metric(example, prediction)

        results.append(
            {
                "research_request": research_request,
                "synthesizer_score": score,
                "synthesized_report": prediction.synthesized_report,
                "gathered_findings": gathered_findings,
            }
        )
        tqdm.write(f"  score: {score:.3f}")

    avg = sum(r["synthesizer_score"] for r in results) / len(results)
    print(f"Synthesizer Avg Score : {avg:.3f}\n")
    return results


def run_annotator_evals(synthesizer_results):
    print("Running Evals for Annotator")
    results = []
    for synth in tqdm(synthesizer_results):
        research_request = synth["research_request"]
        synthesized_report = synth["synthesized_report"]
        gathered_findings = synth["gathered_findings"]

        prediction = annotator(
            synthesized_report=synthesized_report,
            processed_sources=gathered_findings,
        )
        example = dspy.Example(
            synthesized_report=synthesized_report,
            processed_sources=gathered_findings,
        ).with_inputs("synthesized_report", "processed_sources")
        score = annotator_metric(example, prediction)

        results.append(
            {
                "research_request": research_request,
                "annotator_score": score,
                "annotated_report": prediction.annotated_report,
            }
        )
        tqdm.write(f"  score: {score:.3f}")

    avg = sum(r["annotator_score"] for r in results) / len(results)
    print(f"Annotator Avg Score : {avg:.3f}\n")
    return results


async def main():
    with open("datasets/snapshots.json") as f:
        all_snapshots = json.load(f)
    snapshots = [all_snapshots[i] for i in TEST]

    clarifier_results = run_clarifier_evals(snapshots)
    planner_results = run_planner_evals(snapshots)
    processor_results = await run_processor_evals(snapshots)
    synthesizer_results = run_synthesizer_evals(snapshots, processor_results)
    annotator_results = run_annotator_evals(synthesizer_results)

    scores_df = pd.DataFrame(
        [
            {
                "research_request": c["research_request"],
                "clarifier_score": c["clarifier_score"],
                "planner_score": p["planner_score"],
                "processor_relevance_score": pr["processor_relevance_score"],
                "synthesizer_score": s["synthesizer_score"],
                "annotator_score": a["annotator_score"],
            }
            for c, p, pr, s, a in zip(
                clarifier_results,
                planner_results,
                processor_results,
                synthesizer_results,
                annotator_results,
            )
        ]
    )
    scores_df.to_csv("evals/baseline_scores.csv", index=False)
    print("Scores saved to evals/baseline_scores.csv")

    outputs_df = pd.DataFrame(
        [
            {
                "research_request": c["research_request"],
                "clarifying_questions": c["clarifying_questions"],
                "research_topics": p["research_topics"],
                "synthesized_report": s["synthesized_report"],
                "annotated_report": a["annotated_report"],
            }
            for c, p, s, a in zip(
                clarifier_results,
                planner_results,
                synthesizer_results,
                annotator_results,
            )
        ]
    )
    outputs_df.to_csv("evals/component_outputs.csv", index=False)
    print("Outputs saved to evals/component_outputs.csv")


asyncio.run(main())
