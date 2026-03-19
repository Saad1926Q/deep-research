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
    research_task: str = dspy.InputField()
    research_subtask: str = dspy.InputField()
    url: str = dspy.InputField()
    page_content: str = dspy.InputField()
    summary: str = dspy.OutputField()
    relevant_facts: list[str] = dspy.OutputField()


class SynthesizerSignature(dspy.Signature):
    research_request: str = dspy.InputField()
    clarifying_questions_and_answers: list[dict[str, str]] = dspy.InputField()
    research_topics: list[str] = dspy.InputField()
    gathered_findings: list[dict] = dspy.InputField(
        desc="per-subtopic processed sources, each with 'subtopic' and 'sources' (list of {url, summary, relevant_facts})"
    )
    final_report: str = dspy.OutputField()


class AnnotatorSignature(dspy.Signature):
    final_report: str = dspy.InputField()
    processed_sources: list[dict] = dspy.InputField(
        desc="per-subtopic processed sources, each with 'subtopic' and 'sources' (list of {url, summary, relevant_facts})"
    )
    annotated_report: str = dspy.OutputField()
