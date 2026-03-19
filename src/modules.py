import dspy

from src.signatures import (
    AnnotatorSignature,
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
processor = dspy.Predict(UrlProcessorSignature)
synthesizer = dspy.Predict(SynthesizerSignature)
annotator = dspy.Predict(AnnotatorSignature)
