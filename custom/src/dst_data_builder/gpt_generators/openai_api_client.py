"""
OpenAI API Client - Handles OpenAI API communication
"""
import logging
from typing import Optional, Tuple
from openai import AsyncOpenAI


class OpenAIAPIClient:
    """Handles OpenAI API communication and client management."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize OpenAI API client.

        Args:
            api_key: OpenAI API key
            base_url: Optional base URL (e.g., for OpenRouter)
            logger: Optional logger instance
        """
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        self.client = self._create_client()

    def _create_client(self) -> Optional[AsyncOpenAI]:
        """Create and return an AsyncOpenAI client."""
        try:
            if self.base_url:
                return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            return AsyncOpenAI(api_key=self.api_key)
        except Exception:
            msg = "Failed to construct AsyncOpenAI client. Invalid API key provided."
            self.logger.warning(msg)
            return None

    async def generate_completion(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[bool, str]:
        """
        Make API call to generate completion.

        Args:
            prompt: The prompt to send to the API
            model: Model name (e.g., "gpt-4o")
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (success: bool, result: str)
            - On success: (True, raw_api_response_string)
            - On failure: (False, error_message)
        """
        if self.client is None:
            return False, "OpenAI client not initialized. Check API key."

        try:
            self.logger.info(f"Making GPT API call with {len(prompt)} character prompt...")

            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating hierarchical task structures.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            raw_content = response.choices[0].message.content.strip()
            self.logger.info(f"Received response: {len(raw_content)} characters")

            return True, raw_content

        except Exception as e:
            self.logger.exception(f"API call failed: {e}")
            return False, f"api_error: {str(e)}"
