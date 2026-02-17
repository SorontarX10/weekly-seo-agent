from __future__ import annotations

from dotenv import find_dotenv, load_dotenv
from langchain_openai import AzureChatOpenAI

from weekly_seo_agent.config import AgentConfig


try:
    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass


def build_gaia_llm(config: AgentConfig) -> AzureChatOpenAI:
    if not config.gaia_llm_enabled:
        raise RuntimeError(
            "GAIA config missing. Required: GAIA_ENDPOINT, GAIA_API_KEY "
            "(or OPENAI_API_KEY), GAIA_API_VERSION, GAIA_MODEL."
        )

    return AzureChatOpenAI(
        azure_endpoint=config.gaia_endpoint,
        api_key=config.gaia_api_key,
        openai_api_version=config.gaia_api_version,
        azure_deployment=config.gaia_model,
        temperature=config.gaia_temperature,
        max_tokens=max(200, int(config.gaia_max_output_tokens)),
        timeout=max(30, int(config.gaia_timeout_sec)),
        max_retries=max(0, int(config.gaia_max_retries)),
    )
