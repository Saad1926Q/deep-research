import os

import dspy

pipeline_lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
# ideally judge_lm should be a stronger model like gpt-4o for more reliable evaluations,
# but using mini for now to keep costs low while learning
judge_lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
