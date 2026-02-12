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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")


def call_llm(prompt):
    """Call LLM API based on configured provider."""

    if LLM_PROVIDER == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    elif LLM_PROVIDER == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    elif LLM_PROVIDER == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 4000,
                "response_mime_type": "application/json"
            }
        )

        return response.text

    elif LLM_PROVIDER == "groq":
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=4000,
            top_p=1
        )

        return response.choices[0].message.content
    elif LLM_PROVIDER == "cerebras":
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
