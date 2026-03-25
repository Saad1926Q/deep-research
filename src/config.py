import os

import dspy

pipeline_lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
judge_lm = dspy.LM("openai/gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
