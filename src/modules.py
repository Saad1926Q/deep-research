import dspy

from src.signatures import (
    ClarifierSignature,
    GathererSignature,
    PlannerSignature,
    SynthesizerSignature,
    UrlProcessorSignature,
)
from src.tools import internet_search

clarifier = dspy.Predict(ClarifierSignature)
planner = dspy.Predict(PlannerSignature)
gatherer = dspy.ReAct(GathererSignature, tools=[internet_search])
processor = dspy.RLM(UrlProcessorSignature)
synthesizer = dspy.Predict(SynthesizerSignature)
