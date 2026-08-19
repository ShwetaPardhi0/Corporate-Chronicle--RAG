"""
LLM Client — wraps Gemini API (google-genai SDK).
The generator logic is kept here so the pipeline and routes can call it
without knowing which underlying model is used.
"""
import os
from google import genai

from src.prompts.prompt_templates import build_rag_prompt, build_mock_response


class LLMClient:
    """Thin wrapper around the Gemini generative model API.

    Falls back to a structured mock response when no API key is provided,
    making local development possible without credentials.
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            print(
                "[LLM] Warning: GEMINI_API_KEY is not set. "
                "LLMClient will run in mock mode."
            )
            self.client = None
        else:
            print(f"[LLM] Initialized Gemini client with model '{model_name}'")
            self.client = genai.Client(api_key=self.api_key)

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate an answer grounded in the retrieved context chunks.

        Args:
            query: The user's original question.
            context_chunks: Ranked list of retrieved chunk dicts.

        Returns:
            Generated answer string.
        """
        if not self.client:
            return build_mock_response(query, context_chunks)

        prompt = build_rag_prompt(query, context_chunks)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            print(f"[LLM] Error calling Gemini API: {e}")
            return f"Generation error: {str(e)}"
