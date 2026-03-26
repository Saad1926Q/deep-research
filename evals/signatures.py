from enum import Enum

import dspy


class VerdictScore(float, Enum):
    NOT_SATISFIED = 0
    PARTIALLY_SATISFIED = 0.5
    SATISFIED = 1


class EvaluateRubricCriterion(dspy.Signature):
    """Evaluate a research report against a single rubric criterion."""

    research_request: str = dspy.InputField()
    rubric_criterion: str = dspy.InputField(
        desc="the criterion to evaluate the report against"
    )
    annotated_report: str = dspy.InputField()
    reasoning: str = dspy.OutputField(
        desc="specific evidence from the report supporting the verdict"
    )
    verdict_score: VerdictScore = dspy.OutputField(
        desc="1 = satisfied, 0.5 = partially satisfied, 0 = not satisfied"
    )
