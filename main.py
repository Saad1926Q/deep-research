import asyncio
from datetime import datetime

import dspy

from src.config import pipeline_lm
from src.modules import annotator, clarifier, gatherer, planner, synthesizer
from src.utils import process_url

dspy.configure(lm=pipeline_lm)


async def main():
    # Step 1: get research query
    research_request = input("Enter your research query: ").strip()

    # Step 2: clarify
    clarifier_result = clarifier(research_request=research_request)

    # Step 3: ask clarifying questions one by one
    clarifying_questions_and_answers = []
    for question in clarifier_result.clarifying_questions:
        print(f"{question}\n")
        answer = input("Your answer: ").strip()
        clarifying_questions_and_answers.append(
            {"question": question, "answer": answer}
        )

    # Step 4: plan subtopics
    planner_result = planner(
        research_request=research_request,
        clarifying_questions_and_answers=clarifying_questions_and_answers,
        max_num_research_topics=5,
    )
    print(f"Research topics: {planner_result.research_topics}\n")

    # Step 5: gather + process each topic
    gathered_findings = []
    for topic in planner_result.research_topics:
        print(f"\nGathering sources for: {topic}")
        gatherer_result = gatherer(
            research_request=research_request,
            subtopic_to_research=topic,
            num_sources=5,
        )

        sources = await asyncio.gather(
            *[
                process_url(url, research_request, topic)
                for url in gatherer_result.urls_list
            ]
        )
        relevant_sources = [s for s in sources if s["page_is_relevant"]]
        gathered_findings.append({"subtopic": topic, "sources": relevant_sources})

    # Step 6: synthesize
    print("Synthesizing report...\n")
    synthesizer_result = synthesizer(
        research_request=research_request,
        clarifying_questions_and_answers=clarifying_questions_and_answers,
        research_topics=planner_result.research_topics,
        gathered_findings=gathered_findings,
    )

    # Step 7: annotate
    annotator_result = annotator(
        synthesized_report=synthesizer_result.synthesized_report,
        processed_sources=gathered_findings,
    )

    print("\n=== FINAL REPORT ===\n")
    print(annotator_result.annotated_report)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(annotator_result.annotated_report)
    print(f"\nReport saved to {filename}")


asyncio.run(main())
