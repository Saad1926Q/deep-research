import asyncio
import os
from datetime import datetime

import dspy

from src.config import pipeline_lm
from src.modules import clarifier, researcher, synthesizer
from src.utils import internet_search, process_url

dspy.configure(lm=pipeline_lm)

MAX_HOPS = 10


async def run_pipeline(
    eval: bool = False,
    research_request: str | None = None,
    name: str | None = None,
    idx: str | None = None,
    clarifying_questions_and_answers: list | None = None,
):
    if not eval:
        # Step 1: get research query
        research_request = input("Enter your research query: ").strip()

    if not eval:
        # Step 2: clarify
        clarifier_result = clarifier(research_request=research_request.strip())

        # Step 3: ask clarifying questions one by one
        clarifying_questions_and_answers = []
        for question in clarifier_result.clarifying_questions:
            print(f"{question}\n")
            answer = input("Your answer: ").strip()
            clarifying_questions_and_answers.append(
                {"question": question, "answer": answer}
            )

    # Step 4: multi-hop research loop
    accumulated_facts = []
    for hop in range(MAX_HOPS):
        researcher_result = researcher(
            research_request=research_request,
            clarifying_questions_and_answers=clarifying_questions_and_answers,
            accumulated_facts=accumulated_facts,
        )

        if researcher_result.is_done:
            print("Research complete.")
            break

        print(f"\nHop {hop + 1}: {researcher_result.search_query}")

        search_results = internet_search(researcher_result.search_query, max_results=3)
        urls = [r["url"] for r in search_results.get("results", [])]

        processed = await asyncio.gather(
            *[process_url(url, research_request) for url in urls]
        )

        for result in processed:
            if result["page_is_relevant"]:
                accumulated_facts.append({
                    "url": result["url"],
                    "facts": result["relevant_facts"],
                })

    # Step 5: synthesize
    print("\nSynthesizing report...")
    synthesizer_result = synthesizer(
        research_request=research_request,
        clarifying_questions_and_answers=clarifying_questions_and_answers,
        accumulated_facts=accumulated_facts,
    )

    print(synthesizer_result.annotated_report)

    if eval:
        os.makedirs(f"reports/{name}", exist_ok=True)
        filename = f"reports/{name}/{idx}.md"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(synthesizer_result.annotated_report)
    print(f"\nReport saved to {filename}")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
