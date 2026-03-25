import asyncio

import dspy

from src.clients import tavily_client


class AnswerClarifyingQuestion(dspy.Signature):
    """Answer a clarifying question about a research request as a user would."""

    research_request: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


answer_question = dspy.Predict(AnswerClarifyingQuestion)


async def process_url(url: str, research_request: str, topic: str) -> dict:
    from src.modules import processor
    print(f"  Reading: {url}")
    page = await asyncio.to_thread(read_webpage, url)
    page_content = page["results"][0]["raw_content"] if page.get("results") else ""
    processor_result = await processor.aforward(
        research_task=research_request,
        research_subtask=topic,
        url=url,
        page_content=page_content,
    )
    return {
        "url": url,
        "page_is_relevant": processor_result.page_is_relevant,
        "summary": processor_result.summary,
        "relevant_facts": processor_result.relevant_facts,
    }


def read_webpage(url: str) -> dict:
    """Read the content of a URL."""
    response = tavily_client.extract([url])
    return response


def build_planner_example(
    research_request: str, clarifier_result: dspy.Prediction
) -> dspy.Example:
    clarifying_questions_and_answers = [
        {
            "question": q,
            "answer": answer_question(
                research_request=research_request, question=q
            ).answer,
        }
        for q in clarifier_result.clarifying_questions
    ]
    return dspy.Example(
        research_request=research_request,
        clarifying_questions_and_answers=clarifying_questions_and_answers,
    ).with_inputs("research_request", "clarifying_questions_and_answers")
