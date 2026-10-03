"""
Multi-Provider LLM Client
Unifies Google Gemini, Anthropic Claude, DeepSeek and Alibaba Qwen behind a
single `call_llm(...)` function, so the rest of the app never needs to know
which provider is active.

Design:
- Gemini uses the official `google-genai` SDK (already a hard dependency of
  this project).
- Claude, DeepSeek and Qwen are called directly over HTTPS with `requests`
  (already a dependency for the optional image-features branch), which
  avoids pulling in three more SDKs just to send one chat completion.
- Every provider's model name is overridable via an environment variable,
  because hosted LLM vendors periodically retire/rename model versions —
  hardcoding one would silently start failing once it's deprecated.
"""

from typing import Optional, Dict, Any
import os
import requests

REQUEST_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# Provider registry — single source of truth for the sidebar dropdown and
# for picking the right HTTP call below.
# ---------------------------------------------------------------------------
PROVIDERS: Dict[str, Dict[str, str]] = {
    "gemini": {
        "label": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "env_model": "GEMINI_MODEL_NAME",
        "default_model": "gemini-3.7-flash",
        "example_model": "gemini-3.7-pro",
        "docs_url": "https://aistudio.google.com/apikey",
    },
    "claude": {
        "label": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "env_model": "ANTHROPIC_MODEL_NAME",
        "default_model": "claude-sonnet-4-5",
        "example_model": "claude-opus-4-1",
        "docs_url": "https://console.anthropic.com/settings/keys",
    },
    "deepseek": {
        "label": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "env_model": "DEEPSEEK_MODEL_NAME",
        "default_model": "deepseek-chat",
        "example_model": "deepseek-reasoner",
        "docs_url": "https://platform.deepseek.com/api_keys",
    },
    "qwen": {
        "label": "Qwen (Alibaba)",
        "env_key": "QWEN_API_KEY",
        "env_model": "QWEN_MODEL_NAME",
        "default_model": "qwen-plus",
        "example_model": "qwen-max",
        "docs_url": "https://dashscope.console.aliyun.com/apiKey",
    },
}


def get_provider_meta(provider_key: str) -> Dict[str, str]:
    return PROVIDERS.get(provider_key, PROVIDERS["gemini"])


def resolve_model_name(provider_key: str, override: Optional[str] = None) -> str:
    meta = get_provider_meta(provider_key)
    if override:
        return override
    return os.getenv(meta["env_model"], meta["default_model"])


class LLMCallError(Exception):
    """Raised when a live LLM API call fails; the caller falls back to the heuristic advisor."""


def call_llm(
    provider: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> str:
    """
    Sends one chat completion request to the chosen provider and returns the
    assistant's text response. Raises LLMCallError on any failure so callers
    can uniformly fall back to the local heuristic advisor.
    """
    provider = (provider or "gemini").lower()
    model = resolve_model_name(provider, model_name)

    try:
        if provider == "gemini":
            return _call_gemini(api_key, system_prompt, user_prompt, model, temperature)
        elif provider == "claude":
            return _call_claude(api_key, system_prompt, user_prompt, model, temperature)
        elif provider == "deepseek":
            return _call_deepseek(api_key, system_prompt, user_prompt, model, temperature)
        elif provider == "qwen":
            return _call_qwen(api_key, system_prompt, user_prompt, model, temperature)
        else:
            raise LLMCallError(f"Unknown provider: {provider}")
    except LLMCallError:
        raise
    except Exception as e:
        raise LLMCallError(str(e)) from e


# ---------------------------------------------------------------------------
# Google Gemini (official SDK)
# ---------------------------------------------------------------------------
def _call_gemini(api_key: str, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            )
        )
        return response.text
    except Exception:
        # Legacy SDK fallback (older google-generativeai package)
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=api_key)
        gmodel = legacy_genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
        response = gmodel.generate_content(user_prompt)
        return response.text


# ---------------------------------------------------------------------------
# Anthropic Claude (REST — Messages API)
# ---------------------------------------------------------------------------
def _call_claude(api_key: str, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 4096,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        raise LLMCallError(f"Claude API HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    parts = data.get("content", [])
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    if not text:
        raise LLMCallError("Claude API returned an empty response.")
    return text


# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible REST)
# ---------------------------------------------------------------------------
def _call_deepseek(api_key: str, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        raise LLMCallError(f"DeepSeek API HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMCallError(f"Unexpected DeepSeek response shape: {e}")


# ---------------------------------------------------------------------------
# Qwen / Alibaba DashScope (OpenAI-compatible REST)
# ---------------------------------------------------------------------------
def _call_qwen(api_key: str, system_prompt: str, user_prompt: str, model: str, temperature: float) -> str:
    resp = requests.post(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        raise LLMCallError(f"Qwen API HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMCallError(f"Unexpected Qwen response shape: {e}")
