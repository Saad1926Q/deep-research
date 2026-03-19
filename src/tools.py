from src.clients import tavily_client


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
