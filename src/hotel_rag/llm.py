from openai import OpenAI


class OpenAICompatClient:
    """LLM client for any OpenAI-compatible endpoint (9router / omniroute)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model:
            raise ValueError("llm_model is not configured (set LLM_MODEL in .env)")

        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout_seconds,
        )

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
