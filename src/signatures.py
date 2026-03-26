import dspy


class AnswerClarifyingQuestion(dspy.Signature):
    """Answer a clarifying question about a research request as a user would."""

    research_request: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ClarifierSignature(dspy.Signature):
    research_request: str = dspy.InputField()
    clarifying_questions: list[str] = dspy.OutputField()


class ResearcherSignature(dspy.Signature):
    """Given the research request and facts gathered so far, reason about what you know
    and what is still missing, then decide the next search query or whether you have enough."""

    research_request: str = dspy.InputField()
    clarifying_questions_and_answers: list[dict[str, str]] = dspy.InputField()
    accumulated_facts: list[dict] = dspy.InputField(
        desc="facts gathered so far, each with 'url' and 'facts' (list of strings). empty at the start."
    )
    search_query: str = dspy.OutputField(
        desc="the next search query to run. ignored if is_done is True."
    )
    is_done: bool = dspy.OutputField(
        desc="True if the accumulated facts are sufficient to fully address the research request and the clarifying questions and answers, False if more research is needed"
    )


class UrlProcessorSignature(dspy.Signature):
    """Extract all information relevant to the research task from the page content.
    Be thorough - capture specific facts, statistics, dates, names, and examples.
    Do not summarize vaguely; preserve concrete detail that would support a research report."""

    research_request: str = dspy.InputField()
    url: str = dspy.InputField()
    page_content: str = dspy.InputField()
    page_is_relevant: bool = dspy.OutputField(
        desc="the page contains information relevant to the research request"
    )
    relevant_facts: list[str] = dspy.OutputField(
        desc="specific facts, statistics, dates, and named examples extracted from the page"
    )


class SynthesizerSignature(dspy.Signature):
    """Synthesize a comprehensive, well-structured research report from the accumulated facts,
    with inline citations added as you write.

    As you write each claim, fact, or statistic, add an inline [N] citation immediately after it.
    At the end of the report, add a 'References' section listing each source as:
    [1] URL
    [2] URL
    ...

    Use the same number wherever the same source is cited more than once.
    Every specific claim must have a citation. Do not add citations at the end of paragraphs -
    cite each individual claim inline as you write it.

    Include specific facts, statistics, dates, and named examples from the sources, and draw
    connections across topics where relevant. Use markdown with clear section headers.
    Do not pad with vague generalities - every claim should be grounded in the provided facts."""

    research_request: str = dspy.InputField()
    clarifying_questions_and_answers: list[dict[str, str]] = dspy.InputField()
    accumulated_facts: list[dict] = dspy.InputField(
        desc="all gathered facts, each with 'url' and 'facts' (list of strings)"
    )
    annotated_report: str = dspy.OutputField(
        desc="comprehensive markdown report with inline [N] citations after each claim, followed by a numbered References section"
    )
