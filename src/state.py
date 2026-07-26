"""Shared graph state for the agent.

LangGraph passes one state object between nodes. Each node returns a
partial update; fields are merged using their reducers.
"""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Conversation state for the chat agent.

    Inherits ``messages`` from ``MessagesState``, which uses the
    ``add_messages`` reducer so new messages are appended rather than
    replaced. Extend this class with additional fields as needed
    (for example ``summary`` or ``user_id``).
    """
