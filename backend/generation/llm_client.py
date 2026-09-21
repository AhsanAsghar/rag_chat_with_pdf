"""
Calls the OpenAI chat completion endpoint with the assembled
RAG prompt and returns the generated answer text.
"""
from openai import OpenAI
from ..config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def generate_answer(messages: list[dict]) -> str:
    response = _client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.2,  # low temperature: favor grounded, consistent answers over creativity
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()
