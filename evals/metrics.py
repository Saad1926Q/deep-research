import dspy
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback

from evals.signatures import (
    AssessAnnotatorOutput,
    AssessFinalReport,
    AssessQuestion,
    AssessTopicCoverage,
)
from src.config import judge_lm

assess_question = dspy.Predict(AssessQuestion)
assess_topic_coverage = dspy.Predict(AssessTopicCoverage)
assess_final_report = dspy.Predict(AssessFinalReport)
assess_annotator_output = dspy.Predict(AssessAnnotatorOutput)

CLARIFIER_THRESHOLD = 0.7
PLANNER_THRESHOLD = 0.7
PROCESSOR_RELEVANCE_THRESHOLD = 0.8
SYNTHESIZER_THRESHOLD = 0.7
ANNOTATOR_THRESHOLD = 0.7


def clarifier_metric(example: dspy.Example, prediction: dspy.Prediction, trace: DSPyTrace | None = None) -> float | ScoreWithFeedback:
    questions = prediction.clarifying_questions
    if not questions:
        return ScoreWithFeedback(score=0.0, feedback="No clarifying questions were generated.") if trace is not None else 0.0
    with dspy.context(lm=judge_lm):
        results = [
            assess_question(research_request=example.research_request, question=q)
            for q in questions
        ]
    score = sum((r.is_relevant + r.helps_narrow_research) / 2 for r in results) / len(results)

    if trace is not None:
        feedback = "\n".join(f"Q: {q}\n{r.feedback}" for q, r in zip(questions, results))
        return ScoreWithFeedback(score=score, feedback=feedback)
    return score


def planner_metric(example: dspy.Example, prediction: dspy.Prediction, trace: DSPyTrace | None = None) -> float | ScoreWithFeedback:
    with dspy.context(lm=judge_lm):
        result = assess_topic_coverage(
            research_request=example.research_request,
            research_topics=prediction.research_topics,
        )
    score = (result.topics_cover_request + result.topics_are_distinct) / 2

    if trace is not None:
        return ScoreWithFeedback(score=score, feedback=result.feedback)
    return score



def processor_relevance_metric(example: dspy.Example, prediction: dspy.Prediction, trace: DSPyTrace | None = None) -> float | ScoreWithFeedback:
    if example.is_hard_negative:
        score = float(not prediction.page_is_relevant)
        feedback = (
            "Correctly rejected this irrelevant page." if score == 1.0
            else "Failed to reject an irrelevant page - this is a hard negative that should have been marked as not relevant."
        )
    else:
        score = float(prediction.page_is_relevant)
        feedback = (
            "Correctly marked this page as relevant." if score == 1.0
            else "Incorrectly marked a relevant page as not relevant."
        )

    if trace is not None:
        return ScoreWithFeedback(score=score, feedback=feedback)
    return score


def synthesizer_metric(example: dspy.Example, prediction: dspy.Prediction, trace: DSPyTrace | None = None) -> float | ScoreWithFeedback:
    with dspy.context(lm=judge_lm):
        result = assess_final_report(
            research_request=example.research_request,
            research_topics=example.research_topics,
            gathered_findings=example.gathered_findings,
            synthesized_report=prediction.synthesized_report,
        )
    score = (
        result.answers_the_request
        + result.uses_provided_findings
        + result.is_well_structured
        + result.covers_all_topics
    ) / 4

    if trace is not None:
        return ScoreWithFeedback(score=score, feedback=result.feedback)
    return score


def annotator_metric(example: dspy.Example, prediction: dspy.Prediction, trace: DSPyTrace | None = None) -> float | ScoreWithFeedback:
    with dspy.context(lm=judge_lm):
        result = assess_annotator_output(
            synthesized_report=example.synthesized_report,
            annotated_report=prediction.annotated_report,
            processed_sources=example.processed_sources,
        )
    score = (result.citations_are_grounded + result.content_is_preserved) / 2

    if trace is not None:
        return ScoreWithFeedback(score=score, feedback=result.feedback)
    return score
