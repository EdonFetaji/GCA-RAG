from anthropic import Anthropic
from dotenv import load_dotenv
import os

from google import genai
from groq import Groq
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras

# Load environment variables
load_dotenv()

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.1-70b")


def call_llm(prompt):
    """Call LLM API based on configured provider."""
    if LLM_PROVIDER == "cerebras":
        client = Cerebras(
            api_key=os.getenv("CEREBRAS_API_KEY"),
        )
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")
