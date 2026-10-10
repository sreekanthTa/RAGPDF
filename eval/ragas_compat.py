"""Workaround for ragas 0.4.3.

ragas/llms/base.py does, at import time:

    from langchain_community.chat_models.vertexai import ChatVertexAI

but LangChain 1.0 removed that module (VertexAI moved to
langchain-google-vertexai). ragas only uses ChatVertexAI in an isinstance()
check, so we inject a minimal stub for it.

Import this module BEFORE importing ragas. See ragas issues #2741 / #2753.
"""

import sys
import types

_MODULE = "langchain_community.chat_models.vertexai"

if _MODULE not in sys.modules:
    _stub = types.ModuleType(_MODULE)

    class ChatVertexAI:  # noqa: D401 - stub; this project never uses VertexAI
        """Placeholder so ragas's isinstance() list can be built."""

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules[_MODULE] = _stub
