import asyncio

import dspy

from src.clients import tavily_client
from src.signatures import AnswerClarifyingQuestion

answer_question = dspy.Predict(AnswerClarifyingQuestion)


def internet_search(
    query: str, max_results: int = 5, include_raw_content: bool = False
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic="general",
    )


async def process_url(url: str, research_request: str) -> dict:
    from src.modules import processor

    print(f"  Reading: {url}")
    page = await asyncio.to_thread(read_webpage, url)
    page_content = page["results"][0]["raw_content"] if page.get("results") else ""
    processor_result = await processor.aforward(
        research_request=research_request,
        url=url,
        page_content=page_content,
    )
    return {
        "url": url,
        "page_is_relevant": processor_result.page_is_relevant,
        "relevant_facts": processor_result.relevant_facts,
    }


def read_webpage(url: str) -> dict:
    """Read the content of a URL."""
    response = tavily_client.extract([url])
    return response
