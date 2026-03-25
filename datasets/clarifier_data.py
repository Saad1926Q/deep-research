import dspy

examples = [
    dspy.Example(
        research_request="Write a report on how the landscape of NLP has evolved over time"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="Generate a research report on some of the most influential works in cinema"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="Detailed research report on how programming languages and practices have evolved since the inception of computing"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How has the rapid spread of AI in the past few years affected everyday life"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="Detailed report on video codecs - their history, how they work, and how they have evolved"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="Report on the consumption of anime over the years across the globe and how it affects different demographics"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="Who are some of the most influential political figures of the last decade and what impact have they had"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How has the field of computer vision evolved from before the existence of CNNs to today"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="What are the different ideas and perspectives that exist around the concept of AGI"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How has the increased accessibility of information through the internet affected populations worldwide"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How has the acceleration of research across different fields changed since LLMs came into play"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How have Disney movies changed and evolved over the years and what different themes have they explored across different eras"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="Detailed research report on Zionism - how it started, its ideological roots, and how it has evolved"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="What are the common beliefs among atheists and what are the reasons people choose this worldview"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How have open world games like GTA affected populations and culture since their inception"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="What led to the creation of the Rust programming language and how does it approach memory safety"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="Comprehensive research report on global warming - causes, current state, and projected impact"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="What are some of the lesser known hidden gems of literature and what makes them worth reading"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How has Counter-Strike evolved as an influential game over the years and how has its fanbase changed over time"
    ).with_inputs("research_request"),
    dspy.Example(
        research_request="How is piracy perceived by different groups - consumers, creators, corporations, and governments - and what is the overall cultural and economic debate around it"
    ).with_inputs("research_request"),
]
