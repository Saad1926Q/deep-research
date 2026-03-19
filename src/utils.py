from src.clients import tavily_client


def read_webpage(url: str) -> dict:
    """Read the content of a URL."""
    response = tavily_client.extract([url])
    return response
