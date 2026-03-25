import dspy


class AssessQuestion(dspy.Signature):
    """Assess whether a clarifying question is useful for a research task."""

    research_request: str = dspy.InputField()
    question: str = dspy.InputField()
    is_relevant: bool = dspy.OutputField(
        desc="question is relevant to the research request"
    )
    helps_narrow_research: bool = dspy.OutputField(
        desc="question would help narrow down or focus the research in a meaningful way"
    )
    feedback: str = dspy.OutputField(
        desc="specific explanation of why this question is or isn't useful for the research task"
    )


class AssessTopicCoverage(dspy.Signature):
    """Assess whether research topics adequately cover the research request."""

    research_request: str = dspy.InputField()
    research_topics: list[str] = dspy.InputField()
    topics_cover_request: bool = dspy.OutputField(
        desc="topics collectively address the full scope of the research request"
    )
    topics_are_distinct: bool = dspy.OutputField(
        desc="topics are non-overlapping with no redundancy"
    )
    feedback: str = dspy.OutputField(
        desc="specific explanation of what the topics cover well and what aspects of the research request are missing or redundant"
    )


class AssessFinalReport(dspy.Signature):
    """Assess the quality of a synthesized research report."""

    research_request: str = dspy.InputField()
    research_topics: list[str] = dspy.InputField()
    gathered_findings: list[dict] = dspy.InputField()
    synthesized_report: str = dspy.InputField()
    answers_the_request: bool = dspy.OutputField(
        desc="report substantively addresses what was asked"
    )
    uses_provided_findings: bool = dspy.OutputField(
        desc="report draws from the gathered findings"
    )
    is_well_structured: bool = dspy.OutputField()
    covers_all_topics: bool = dspy.OutputField(
        desc="report has substantive coverage of every research topic"
    )
    feedback: str = dspy.OutputField(
        desc="specific explanation of what the report did well and what it failed at, with concrete examples from the report content"
    )


class AssessAnnotatorOutput(dspy.Signature):
    """Assess the quality of citations added to a research report."""

    synthesized_report: str = dspy.InputField(desc="original report before annotation")
    annotated_report: str = dspy.InputField(desc="report with citations added")
    processed_sources: list[dict] = dspy.InputField(
        desc="per-subtopic sources, each with 'subtopic' and 'sources' (list of {url, summary, relevant_facts})"
    )
    citations_are_grounded: bool = dspy.OutputField(
        desc="every citation in the annotated report actually appears in processed_sources - no hallucinated urls or sources"
    )
    content_is_preserved: bool = dspy.OutputField(
        desc="annotated report contains the same content as the original, only citations were added"
    )
    feedback: str = dspy.OutputField(
        desc="specific explanation of which citations are hallucinated or missing, and whether any report content was changed or dropped"
    )
