# Deep Research

An attempt to implement deep research using DSPy - and in the process, learn about things like evals, prompt optimization, and how these kinds of systems actually work under the hood.

## What this is

basically a research agent that you give a query, it asks some clarifying questions, breaks it into subtopics, searches the web, reads pages, and writes a report with citations. but the main point isn't really just building something that works - it's going through the whole loop of building, evaluating, and actually optimizing an LLM pipeline using DSPy and learning stuff along the way.

## Architecture

```
User Query
    → Clarifier      asks clarifying questions to narrow the scope
    → Planner        breaks the query into research subtopics
    → Gatherer       finds relevant URLs per subtopic (web search)
    → Processor      reads each page and extracts facts and summaries
    → Synthesizer    writes a structured report from all the findings
    → Annotator      adds inline citations to the report
```

Each step is a DSPy module. One thing worth mentioning is the Processor uses `dspy.RLM` - RLM stands for Retrieve-Language Model. The intuition here is that we're passing in the full content of a webpage, which can get really long and blow up the context window. instead of compacting it or truncating it, it makes more sense to treat the page content as an external store that the LM retrieves from - so the LM only actually sees the relevant chunks, not the whole thing. that's basically what RLM does.

## Approach

The rough idea was:

1. Build a base implementation that works end to end
2. Run evals on the baseline
3. Optimize based on eval results using GEPA
4. Re-run evals and compare

## Evals

Instead of just eyeballing outputs and guessing whether the pipeline is doing well, I want to actually quantitatively measure how each component is performing. So the plan is to run evals on each module individually and get concrete scores I can track and optimize against.

### Why I freeze eval data

Web search is non-deterministic - the same query on different days returns different URLs, and page content changes. If evals hit the live web every run, score differences could come from changed search results rather than anything the optimization did.

So I freeze the web-facing part of the pipeline (gatherer + page fetch) into a snapshot collected once. All eval runs draw from the same frozen page content, so any score difference is attributable to the modules being evaluated, not the data.

### What I evaluate

**Clarifier** - questions are relevant and help narrow the research request

**Planner** - topics cover the full scope, are distinct and non-overlapping

**Processor** - correctly rejects irrelevant pages (hard negatives)

I wanted to also evaluate whether the extracted facts are grounded in the actual page content (no hallucinations), but couldn't find a clean way to do this. Passing full raw page content to a judge LM is expensive at eval scale, and the approaches I tried weren't reliable enough to be worth it.

**Synthesizer** - answers the request, uses the gathered findings, well-structured, doesn't leak content from irrelevant pages

**Annotator** - citations are grounded in actual sources, report content is unchanged, follows `[N]` format with a References section

### Data splits

20 snapshots split into:
- **Test (6)** - held out, used only for baseline vs optimized comparison
- **Train (4)** - used by GEPA to bootstrap traces and few-shot examples
- **Val (10)** - used by GEPA to evaluate candidate prompts during optimization

The dataset is small - I'm aware of that. Collecting snapshots is manual work (run the full pipeline, then manually add hard negative sources per topic), and I don't want to spend a ton of time and money just on data collection at this stage.

---

## Notes

### 25th March, 2026

I thought the implementation was basically done and I just needed to run evals and then optimize. Turns out I was wrong.

After running evals I found a problem with the architecture itself. The annotator was scoring low, and after thinking about it I realized the issue: the annotator has no idea which source belongs to which section of the report. The synthesizer takes all the findings and writes a report, but the connection between claims and sources is lost in that step. So when the annotator tries to add citations, it's essentially guessing - which is why it hallucinates.

The fix is to merge the synthesizer and annotator into a single step, since the synthesizer is the only point in the pipeline where the source-to-claim mapping is actually known.

The other thing I realized is that the research feels quite rigid. It plans N topics upfront, gathers N sources per topic, and that's it. It can't follow up on something interesting it found, or adjust what it searches for based on what it already knows. That's a big gap compared to how Perplexity or Claude's deep research actually works - those are loops, not pipelines. The agent decides what to search for next based on what it just read, which is fundamentally different.

Looking into this led me to multi-hop retrieval, which is the pattern that actually makes this kind of dynamic research possible. Might explore that next.

So two main takeaways from this initial exploration:
1. The synthesizer and annotator need to be merged - the architecture breaks the source tracking
2. The pipeline is too rigid - multi-hop is the right direction for more autonomous research
