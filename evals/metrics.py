import dspy
from dspy.teleprompt.gepa.gepa_utils import DSPyTrace, ScoreWithFeedback

from evals.signatures import EvaluateRubricCriterion
from src.config import judge_lm

evaluate_rubric_criterion = dspy.Predict(EvaluateRubricCriterion)


def rubric_metric(
    example: dspy.Example,
    prediction: dspy.Prediction | None,
    trace: DSPyTrace | None = None,
) -> float | ScoreWithFeedback:
    report = example.report if prediction is None else prediction.annotated_report
    with dspy.context(lm=judge_lm):
        results = [
            evaluate_rubric_criterion(
                research_request=example.research_request,
                rubric_criterion=r["criterion"],
                annotated_report=report,
            )
            for r in example.rubrics
        ]

    numerator = sum(
        result.verdict_score.value * r["weight"]
        for result, r in zip(results, example.rubrics)
    )
    denominator = sum(r["weight"] for r in example.rubrics if r["weight"] > 0)
    score = numerator / denominator if denominator > 0 else 0.0

    if trace is not None:
        feedback = "\n".join(
            f"[w={r['weight']}] {r['criterion']}\n{result.reasoning}"
            for r, result in zip(example.rubrics, results)
        )
        return ScoreWithFeedback(score=score, feedback=feedback)

    return score
