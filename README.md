# Deep Research

An attempt to implement deep research using DSPy - and in the process, learn about things like evals, prompt optimization, and how these kinds of systems actually work under the hood.

## What this is

basically a research agent that you give a query, it asks some clarifying questions, breaks it into subtopics, searches the web, reads pages, and writes a report with citations. but the main point isn't really just building something that works - it's going through the whole loop of building, evaluating, and actually optimizing an LLM pipeline using DSPy and learning stuff along the way.

## Architecture

```
User Query
    → Clarifier      asks clarifying questions to narrow the scope

    → Multi-hop retrieval loop
        at each hop:
            - takes accumulated facts (empty initially) + query
            - generates a chain of thought: "I know X, but I still need Y"
            - uses that CoT to fire a web search
            - fetches the page, runs it through the processor, adds facts to the pool
            - also generates a boolean: is retrieval done?
            - stops when done=True or max hops is reached

    → Synthesizer    takes the accumulated fact pool, writes report with inline citations
```

Each step is a DSPy module. One thing worth mentioning is the Processor uses `dspy.RLM` - RLM stands for Retrieve-Language Model. The intuition here is that we're passing in the full content of a webpage, which can get really long and blow up the context window. instead of compacting it or truncating it, it makes more sense to treat the page content as an external store that the LM retrieves from - so the LM only actually sees the relevant chunks, not the whole thing. that's basically what RLM does.

## Approach

The rough idea was:

1. Build a base implementation that works end to end
2. Run evals on the baseline
3. Optimize based on eval results using GEPA
4. Re-run evals and compare

The next direction is replacing the rigid plan-then-gather pipeline with a multi-hop retrieval loop. Instead of planning N topics upfront and searching for each, the agent dynamically decides what to search for at each hop based on what it's already learned:

- at each hop, a researcher component looks at the research request + accumulated facts and reasons about what it knows and what's still missing (chain of thought style)
- it then generates the next search query and a `is_done` signal
- the search returns 3 URLs, all read in parallel, facts added to the pool
- loop continues until `is_done` or max hops reached
- synthesizer then writes the report from the full fact pool

this is closer to how real deep research works — the agent adapts its search strategy as it learns, rather than committing to a fixed plan upfront.

## Evals

Instead of just eyeballing outputs and guessing whether the pipeline is doing well, I want to actually quantitatively measure how the system is performing. The approach is rubric-based evaluation - inspired by [ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents](https://arxiv.org/abs/2511.07685) (Scale AI, 2025), which was built specifically for evaluating deep research agents. we use their [dataset](https://huggingface.co/datasets/ScaleAI/researchrubrics) directly — the prompts and rubrics are already written, so we just run the pipeline on each prompt and score the output. that said, we only evaluate on a small subset (5 samples) given the limited budget — each run costs API calls for both the pipeline and the judge. the goal here isn't to produce state-of-the-art numbers anyway, it's to go through the full loop of building, evaluating, and optimizing an LLM pipeline and actually understand how these ideas work in practice.

### How rubric-based evals work

Each research prompt comes paired with a rubric - a list of specific criteria the response should satisfy, each with a weight. The judge LM scores each criterion:

- `1` — satisfied
- `0.5` — partially satisfied
- `0` — not satisfied

The final compliance score is:

```
score = sum(verdict × weight for all criteria) / sum of positive weights
```

Weights reflect importance:
- `[+4, +5]` — critically important, required for a minimally viable response. without this the response is fundamentally flawed.
- `[+2, +3]` — important, a key feature of a strong response but not absolutely essential.
- `+1` — slightly important, a nice-to-have that improves a good response but doesn't significantly change overall quality.
- `-1` — slightly detrimental, a minor issue or stylistic weakness that doesn't impact core quality.
- `[-3, -2]` — detrimental, a significant error that hurts quality or introduces faulty logic.
- `[-5, -4]` — critically detrimental, so severe it makes the response actively harmful or completely invalidates its reasoning.

this is much more meaningful than generic LLM judge questions because the criteria are specific to each prompt. for example, given a prompt like "write a history of Counter-Strike as an esport", a generic judge asking "is this well structured?" tells you very little. but a rubric criterion like "does the response cover the transition from CS 1.6 to CS:GO to CS2?" actually measures whether the report did its job.

### Data

Using a subset of prompts from the ResearchRubrics dataset. the rubrics are already written. running the pipeline on each prompt produces a markdown report, and the evaluator scores it against the rubric automatically.

### What I evaluate

we'll be evaluating the final report generated by the pipeline using this rubric-based approach.

### Running evals

First generate reports by running the pipeline on each sample, then evaluate:

```bash
python -m evals.run_evals --name baseline
```

Expects reports at `reports/baseline/<sample_id>.md`. Saves scores to `evals/baseline_results.csv`.

---
