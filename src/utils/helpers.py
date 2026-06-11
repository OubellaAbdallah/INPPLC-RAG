"""Utility helper functions for the INPPLC application.

Provides chat history formatting and other shared utilities.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage


def format_chat_history_for_prompt(
    chat_history: list[HumanMessage | AIMessage],
    max_turns: int = 10,
) -> str:
    """Format chat history into a string for the LLM prompt.

    Excludes the most recent message (which is the current query) and
    caps history at max_turns to prevent context window overflow.

    Args:
        chat_history: List of LangChain message objects.
        max_turns: Maximum number of message pairs to include.

    Returns:
        Formatted string of conversation history.
    """
    history_to_format = chat_history[:-1]

    if max_turns > 0:
        history_to_format = history_to_format[-(max_turns * 2):]

    formatted_lines: list[str] = []
    for msg in history_to_format:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"User Question: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"INPPLC Assistant Response: {msg.content}")

    return "\n".join(formatted_lines)
