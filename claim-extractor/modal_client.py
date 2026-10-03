"""Client for the DeepSeek V4.1 Flash Modal OpenAI-compatible endpoint."""

from __future__ import annotations

import json
from typing import TypeVar

import requests
from pydantic import BaseModel


DEFAULT_MODAL_ENDPOINT = (
    "https://ashmanm--ep-deepseek-v4-1-flash-server.us-west.modal.direct"
    "/v1/chat/completions"
)
DEFAULT_MODAL_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModalAPIError(RuntimeError):
    def __init__(self, status_code: int | None, message: str) -> None:
        self.status_code = status_code
        super().__init__(message)


def _extract_streamed_content(response: requests.Response) -> str:
    chunks: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModalAPIError(
                response.status_code,
                f"Invalid SSE event from Modal: {line}",
            ) from exc
        if "error" in event:
            raise ModalAPIError(
                response.status_code,
                f"Modal streaming error: {json.dumps(event['error'], ensure_ascii=False)}",
            )
        try:
            choice = event["choices"][0]
        except (KeyError, IndexError, TypeError):
            continue
        delta = choice.get("delta", {})
        content = delta.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        message_content = choice.get("message", {}).get("content")
        if isinstance(message_content, str):
            chunks.append(message_content)
    return "".join(chunks)


def generate_structured(
    *,
    endpoint: str,
    token_id: str,
    token_secret: str,
    model: str,
    prompt: str,
    schema: type[SchemaT],
    timeout_seconds: float = 600,
) -> SchemaT:
    json_schema = schema.model_json_schema()
    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token_id}.{token_secret}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a concise technical assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
                "top_p": 0.9,
                "stream": True,
                "reasoning_effort": "none",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__.lower(),
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            },
            stream=True,
            timeout=(15, timeout_seconds),
        )
    except requests.RequestException as exc:
        raise ModalAPIError(None, f"Modal connection error at {endpoint}: {exc}") from exc

    if not response.ok:
        try:
            detail = json.dumps(response.json(), ensure_ascii=False)
        except ValueError:
            detail = response.text
        raise ModalAPIError(
            response.status_code,
            f"Modal HTTP {response.status_code}: {detail}",
        )

    content = _extract_streamed_content(response)
    if not content.strip():
        raise ModalAPIError(
            response.status_code,
            "Modal returned no assistant content in the SSE stream.",
        )
    return schema.model_validate_json(content)
