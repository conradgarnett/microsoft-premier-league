"""Chat clients for the agents.

* Azure OpenAI / Microsoft Foundry model deployments (``AZURE_OPENAI_ENDPOINT`` +
  ``AZURE_OPENAI_DEPLOYMENT`` with ``AZURE_OPENAI_API_KEY`` or Entra ID via
  ``AZURE_OPENAI_USE_ENTRA=1``) through Agent Framework's OpenAI chat client.
* ``OfflineChatClient``: deterministic, template-based, used automatically when no
  credentials are configured (and as the fallback when a model call fails validation).
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent_framework import BaseChatClient, ChatResponse, Message

from . import offline


class OfflineChatClient(BaseChatClient):
    """An Agent Framework chat client that answers JSON task requests with templates."""

    OTEL_PROVIDER_NAME = "offline"

    async def _inner_get_response(self, *, messages, stream, options, **kwargs) -> Any:
        request = json.loads(messages[-1].text)
        text = offline.run_task(request)
        if stream:
            from agent_framework import ChatResponseUpdate

            async def _stream():
                yield ChatResponseUpdate(role="assistant", contents=[{"type": "text", "text": text}])

            return _stream()
        return ChatResponse(messages=[Message(role="assistant", contents=[text])], response_id="offline")


def provider_name() -> str:
    return "azure-openai" if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_DEPLOYMENT") else "offline"


def make_chat_client() -> BaseChatClient:
    if provider_name() == "offline":
        return OfflineChatClient()
    from agent_framework.openai import OpenAIChatCompletionClient

    kwargs: dict[str, Any] = {
        "model": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    }
    if os.getenv("AZURE_OPENAI_USE_ENTRA") == "1":
        from azure.identity import DefaultAzureCredential

        kwargs["credential"] = DefaultAzureCredential()
    else:
        kwargs["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
    return OpenAIChatCompletionClient(**kwargs)
