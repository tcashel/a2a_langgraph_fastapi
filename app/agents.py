import re
from typing import List, Tuple

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool


def _llm():
    # Small, affordable default; override with env if desired
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_echo_agent():
    """
    A simple conversational agent that *repeats* the user input and
    briefly explains what it will do. We bias with system instructions.
    """
    system = (
        "You are EchoAgent. Always repeat the user's last message verbatim, "
        "then add one short sentence explaining that you echoed it."
    )
    llm = _llm()
    # No external tools needed; create_react_agent can run tool-less.
    return create_react_agent(
        llm,
        tools=[],
        state_modifier=SystemMessage(system),
    )


@tool
def sum_numbers(text: str) -> str:
    """Extract numbers from the text and return their sum with a short explanation."""
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]
    total = sum(nums) if nums else 0.0
    return f"Detected numbers: {nums}. Sum = {total:g}."


def build_math_agent():
    """
    Detect numbers in the message and return a sum + explanation.
    The tool guarantees deterministic arithmetic; the LLM decides to call it.
    """
    system = (
        "You are MathAgent. If the user message contains numbers, call the "
        "`sum_numbers` tool exactly once using the user message as input, then respond "
        "with the tool output. If there are no numbers, say so briefly."
    )
    llm = _llm()
    return create_react_agent(
        llm,
        tools=[sum_numbers],
        state_modifier=SystemMessage(system),
    )
