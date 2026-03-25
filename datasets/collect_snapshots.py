import asyncio
import json

import dspy

from datasets.clarifier_data import examples
from src.config import pipeline_lm
from src.modules import gatherer, planner
from src.utils import answer_question, read_webpage

dspy.configure(lm=pipeline_lm)


async def fetch_topic_sources(research_request: str, topic: str) -> dict:
    gatherer_result = gatherer(
        research_request=research_request,
        subtopic_to_research=topic,
        num_sources=5,
    )

    async def fetch_url(url: str) -> dict:
        page = await asyncio.to_thread(read_webpage, url)
        page_content = page["results"][0]["raw_content"] if page.get("results") else ""
        return {"url": url, "page_content": page_content, "is_hard_negative": False}

    sources = await asyncio.gather(
        *[fetch_url(url) for url in gatherer_result.urls_list]
    )
    return {"topic": topic, "sources": list(sources)}


async def collect_snapshot(research_request: str) -> dict:
    print(f"\nCollecting: {research_request[:60]}...")

    clarifier_result = dspy.Predict(
        "research_request -> clarifying_questions: list[str]"
    )(research_request=research_request)

    clarifying_questions_and_answers = [
        {
            "question": q,
            "answer": answer_question(
                research_request=research_request, question=q
            ).answer,
        }
        for q in clarifier_result.clarifying_questions
    ]

    planner_result = planner(
        research_request=research_request,
        clarifying_questions_and_answers=clarifying_questions_and_answers,
        max_num_research_topics=5,
    )

    topic_sources = await asyncio.gather(
        *[
            fetch_topic_sources(research_request, topic)
            for topic in planner_result.research_topics
        ]
    )

    return {
        "research_request": research_request,
        "clarifying_questions_and_answers": clarifying_questions_and_answers,
        "research_topics": planner_result.research_topics,
        "topic_sources": list(topic_sources),
    }


async def main():
    snapshots = []
    for example in examples:
        snapshot = await collect_snapshot(example.research_request)
        snapshots.append(snapshot)

    with open("datasets/snapshots.json", "w") as f:
        json.dump(snapshots, f)

    print(f"\nSaved {len(snapshots)} snapshots to datasets/snapshots.json")


asyncio.run(main())
