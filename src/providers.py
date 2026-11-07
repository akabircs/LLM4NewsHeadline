\
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 50


class TextGenerator:
    def generate(self, prompt: str) -> str:  # pragma: no cover - external API
        raise NotImplementedError


class GeminiLangChainGenerator(TextGenerator):
    def __init__(self, model: str, config: DecodingConfig):
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.model = ChatGoogleGenerativeAI(
            model=model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_output_tokens,
            google_api_key=key,
        )

    def generate(self, prompt: str) -> str:
        result = self.model.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)


class TogetherGenerator(TextGenerator):
    def __init__(self, model: str, config: DecodingConfig):
        from langchain_together import ChatTogether
        key = os.getenv("TOGETHER_API_KEY")
        if not key:
            raise RuntimeError("TOGETHER_API_KEY is not set")
        self.model = ChatTogether(
            model=model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_output_tokens,
            together_api_key=key,
        )

    def generate(self, prompt: str) -> str:
        result = self.model.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)


class OpenAIGenerator(TextGenerator):
    def __init__(self, model: str, config: DecodingConfig):
        from langchain_openai import ChatOpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = ChatOpenAI(
            model=model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_output_tokens,
            api_key=key,
        )

    def generate(self, prompt: str) -> str:
        result = self.model.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)


class GeminiDirectGenerator(TextGenerator):
    """Direct Google GenAI client used for the Shironaam experiment."""
    def __init__(self, model: str, config: DecodingConfig):
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model_name = model
        self.config = config

    def generate(self, prompt: str) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=self.config.max_output_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            ),
        )
        return response.text or ""


def make_generator(provider: str, config: DecodingConfig) -> TextGenerator:
    provider = provider.lower()
    if provider == "gemini":
        return GeminiLangChainGenerator("gemini-2.0-flash", config)
    if provider == "llama":
        return TogetherGenerator("meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", config)
    if provider == "gpt4o":
        return OpenAIGenerator("gpt-4o", config)
    raise ValueError(f"Unsupported provider: {provider}")
