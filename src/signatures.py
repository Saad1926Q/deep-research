import dspy


class ClarifierSignature(dspy.Signature):
    research_request: str = dspy.InputField()
    clarifying_questions: list[str] = dspy.OutputField()


class PlannerSignature(dspy.Signature):
    research_request: str = dspy.InputField()
    clarifying_questions_and_answers: list[dict[str, str]] = dspy.InputField()
    max_num_research_topics: int = dspy.InputField()
    research_topics: list[str] = dspy.OutputField()


class GathererSignature(dspy.Signature):
    research_request: str = dspy.InputField()
    subtopic_to_research: str = dspy.InputField()
    num_sources: int = dspy.InputField()
    urls_list: list[str] = dspy.OutputField(desc="relevant urls to investigate")


class UrlProcessorSignature(dspy.Signature):
    """Extract all information relevant to the research task from the page content.
    Be thorough - capture specific facts, statistics, dates, names, and examples.
    Do not summarize vaguely; preserve concrete detail that would support a research report."""

    research_task: str = dspy.InputField()
    research_subtask: str = dspy.InputField()
    url: str = dspy.InputField()
    page_content: str = dspy.InputField()
    page_is_relevant: bool = dspy.OutputField(desc="the page contains information relevant to the research task and subtask")
    summary: str = dspy.OutputField(desc="detailed summary of the page's relevance to the research task")
    relevant_facts: list[str] = dspy.OutputField(desc="specific facts, statistics, dates, and named examples extracted from the page")


class SynthesizerSignature(dspy.Signature):
    """Synthesize a comprehensive, well-structured research report from the gathered findings.
    The report must cover every research topic in depth, include specific facts, statistics, dates,
    and named examples from the sources, and draw connections across topics where relevant.
    Use markdown with clear section headers. Do not pad with vague generalities - every claim
    should be grounded in the provided findings."""

    research_request: str = dspy.InputField()
    clarifying_questions_and_answers: list[dict[str, str]] = dspy.InputField()
    research_topics: list[str] = dspy.InputField()
    gathered_findings: list[dict] = dspy.InputField(
        desc="per-subtopic processed sources, each with 'subtopic' and 'sources' (list of {url, summary, relevant_facts})"
    )
    synthesized_report: str = dspy.OutputField(
        desc="comprehensive markdown report with a section per research topic, grounded in specific facts and examples from the sources"
    )


class AnnotatorSignature(dspy.Signature):
    """Annotate the report with numbered inline citations in the format [1], [2], etc.
    At the end of the report, add a 'References' section listing each source as:
    [1] URL
    [2] URL
    ...
    Every specific claim, fact, or statistic in the report must have an inline citation.
    Use the same number wherever the same source is cited more than once."""

    synthesized_report: str = dspy.InputField()
    processed_sources: list[dict] = dspy.InputField(
        desc="per-subtopic processed sources, each with 'subtopic' and 'sources' (list of {url, summary, relevant_facts})"
    )
    annotated_report: str = dspy.OutputField(
        desc="the report with inline [N] citations after each claim, followed by a numbered References section"
    )
