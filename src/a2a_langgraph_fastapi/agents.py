import re
from typing import List, Tuple

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver


def _llm():
    # Small, affordable default; override with env if desired
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_echo_agent():
    """
    A conversational agent that remembers information shared in the conversation
    and can engage in multi-turn dialogue while still echoing user messages.
    """
    system = (
        "You are EchoAgent, a friendly conversational assistant. You have two main behaviors:\n"
        "1. You always repeat the user's last message verbatim, then add a brief explanation.\n"
        "2. You remember information shared throughout the conversation and can reference it later.\n\n"
        "For example, if someone tells you their name is Dave, you should remember that. "
        "If they later ask 'What's my name?', you should be able to say 'Your name is Dave' "
        "based on what they told you earlier in our conversation.\n\n"
        "Always be conversational and friendly, and use the information you've learned "
        "about the user to provide more personalized responses."
    )
    llm = _llm()
    # No external tools needed; create_react_agent can run tool-less.
    agent = create_react_agent(
        llm,
        tools=[],
        prompt=SystemMessage(system),
        checkpointer=MemorySaver(),
    )
    return agent


@tool
def sum_numbers(text: str) -> str:
    """Extract numbers from the text and return their sum with a short explanation."""
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]
    total = sum(nums) if nums else 0.0
    return f"Detected numbers: {nums}. Sum = {total:g}."


def build_math_agent():
    """
    A conversational math agent that can remember information shared in conversation
    and help with math problems while being friendly and engaging.
    """
    system = (
        "You are MathAgent, a friendly and helpful math assistant. You have two main capabilities:\n"
        "1. You can help with math problems using the sum_numbers tool when users mention numbers.\n"
        "2. You remember information shared throughout the conversation and can reference it later.\n\n"
        "For example, if someone tells you their name is Alice, you should remember that. "
        "If they later ask 'What's my name?', you should be able to say 'Your name is Alice' "
        "based on what they told you earlier in our conversation.\n\n"
        "When users mention numbers, use the sum_numbers tool to calculate their sum. "
        "Always be conversational and friendly, and use the information you've learned "
        "about the user to provide more personalized responses.\n\n"
        "If there are no numbers in the message, respond conversationally and helpfully."
    )
    llm = _llm()
    agent = create_react_agent(
        llm,
        tools=[sum_numbers],
        prompt=SystemMessage(system),
        checkpointer=MemorySaver(),
    )
    return agent
