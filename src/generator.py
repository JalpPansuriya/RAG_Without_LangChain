"""
Answer Generator Module for RAG from Scratch.
Builds the prompt and interfaces with the OpenAI Chat Completion API to generate answers.
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Logger configuration
logger = logging.getLogger(__name__)

# Check API key presence and print warning if missing
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("WARNING: GROQ_API_KEY environment variable is missing.")
    logger.warning("GROQ_API_KEY environment variable is missing.")

def get_client() -> OpenAI:
    """
    Returns an initialized OpenAI client pointing to the Groq base URL.
    Raises ValueError if the API key is missing.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Cannot initialize OpenAI client for Groq: GROQ_API_KEY is missing.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")

def generate(query: str, context_chunks: list[dict]) -> str:
    """
    Constructs a context-grounded prompt and queries OpenAI's gpt-4o-mini
    to generate an answer strictly limited to the provided context.
    
    Args:
        query (str): The user's question.
        context_chunks (list[dict]): List of retrieved chunk dictionaries.
        
    Returns:
        str: The generated answer or a fallback string if no context is available.
    """
    fallback_message = "I don't know based on the provided context."
    
    if not context_chunks:
        return fallback_message
        
    # Get initialized OpenAI client
    client = get_client()
    
    # Format the context block: [i] text (source: filename)
    context_lines = []
    for idx, chunk in enumerate(context_chunks, 1):
        text = chunk.get("text", "").strip()
        source = chunk.get("source", "Unknown")
        context_lines.append(f"[{idx}] {text}  (source: {source})")
        
    context_block = "\n".join(context_lines)
    
    # Build prompt template using plain Python f-strings
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the context provided below.\n"
        "If the answer is not in the context, say \"I don't know based on the provided context.\"\n"
        "Do not make up information."
    )
    
    user_prompt = f"Context:\n{context_block}\n\nUser Question:\n{query}"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0  # Set temperature to 0 for most deterministic grounding
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling OpenAI Chat Completion: {e}")
        raise e
