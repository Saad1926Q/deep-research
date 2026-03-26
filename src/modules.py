import dspy
from dspy.signatures.field import OutputField

from src.signatures import (
    ClarifierSignature,
    ResearcherSignature,
    SynthesizerSignature,
    UrlProcessorSignature,
)

clarifier = dspy.Predict(ClarifierSignature)
researcher = dspy.ChainOfThought(
    ResearcherSignature,
    rationale_field=OutputField(
        desc="reason about what you already know and what is still missing - 'I know X, but I still need Y'"
    ),
)
processor = dspy.RLM(UrlProcessorSignature)
synthesizer = dspy.Predict(SynthesizerSignature)
