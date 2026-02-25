from weekly_seo_agent.config import AgentConfig
from weekly_seo_agent.weekly_reporting_agent.main import (
    QUALITY_MAX_GAIA_MODEL,
    _apply_weekly_quality_max_profile,
)


def test_gsc_country_filter_maps_pl_to_pol(monkeypatch):
    monkeypatch.setenv("GSC_COUNTRY_FILTER", "PL")
    config = AgentConfig.from_env()
    assert config.gsc_country_filter == "pol"


def test_gsc_country_filter_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GSC_COUNTRY_FILTER", "none")
    config = AgentConfig.from_env()
    assert config.gsc_country_filter == ""


def test_placeholder_value_falls_back_to_target(monkeypatch):
    monkeypatch.setenv("TARGET_DOMAIN", "https://allegro.pl")
    monkeypatch.setenv("GSC_SITE_URL", "GSC_SITE_URL=")
    config = AgentConfig.from_env()
    assert config.gsc_site_url == "https://allegro.pl/"


def test_product_trend_defaults(monkeypatch):
    monkeypatch.delenv("PRODUCT_TRENDS_ENABLED", raising=False)
    monkeypatch.delenv("PRODUCT_TRENDS_TOP_ROWS", raising=False)
    monkeypatch.delenv("PRODUCT_TRENDS_HORIZON_DAYS", raising=False)
    config = AgentConfig.from_env()
    assert config.product_trends_enabled is True
    assert config.product_trends_top_rows == 12
    assert config.product_trends_horizon_days == 31


def test_report_country_defaults(monkeypatch):
    monkeypatch.delenv("REPORT_COUNTRIES", raising=False)
    config = AgentConfig.from_env()
    assert config.report_countries == ("PL", "CZ", "SK", "HU")


def test_senuto_country_id_map_from_env(monkeypatch):
    monkeypatch.setenv("SENUTO_COUNTRY_ID_MAP", "PL:1,CZ:50,SK:164,HU:82")
    config = AgentConfig.from_env()
    assert config.senuto_country_id_map["PL"] == 1
    assert config.senuto_country_id_map["CZ"] == 50


def test_ga4_property_id_map_from_env(monkeypatch):
    monkeypatch.setenv("GA4_PROPERTY_ID_MAP", "PL:111,CZ:222,SK:333,HU:444")
    config = AgentConfig.from_env()
    assert config.ga4_property_id_map["PL"] == "111"
    assert config.ga4_property_id_map["CZ"] == "222"


def test_country_specific_external_maps_defaults(monkeypatch):
    monkeypatch.delenv("WEATHER_LATITUDE_MAP", raising=False)
    monkeypatch.delenv("HOLIDAYS_COUNTRY_CODE_MAP", raising=False)
    monkeypatch.delenv("GOOGLE_TRENDS_RSS_URL_MAP", raising=False)
    config = AgentConfig.from_env()
    assert config.weather_latitude_map["PL"] > 50.0
    assert config.weather_latitude_map["CZ"] > 49.0
    assert config.holidays_country_code_map["HU"] == "HU"
    assert "geo=SK" in config.google_trends_rss_url_map["SK"]


def test_allegro_trends_config(monkeypatch):
    monkeypatch.setenv("ALLEGRO_TRENDS_ENABLED", "true")
    monkeypatch.setenv("ALLEGRO_TRENDS_BASIC_AUTH_LOGIN", "search-trends-ui")
    monkeypatch.setenv("ALLEGRO_TRENDS_BASIC_AUTH_PASSWORD", "basic_secret")
    monkeypatch.setenv("ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_LOGIN", "seo_allegro")
    monkeypatch.setenv("ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_PASSWORD", "tech_secret")
    monkeypatch.setenv("ALLEGRO_TRENDS_MEASURES", "VISIT,PV,GMV")
    monkeypatch.setenv("ALLEGRO_TRENDS_TOP_ROWS", "12")

    config = AgentConfig.from_env()

    assert config.allegro_trends_enabled is True
    assert config.allegro_trends_api_enabled is True
    assert config.allegro_trends_measures == ("VISIT", "PV", "GMV")
    assert config.allegro_trends_top_rows == 12


def test_weekly_quality_max_profile_forces_llm_settings(monkeypatch):
    monkeypatch.setenv("GAIA_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_MAP_MAX_TOKENS", "300")
    monkeypatch.setenv("LLM_REDUCE_MAX_TOKENS", "900")
    monkeypatch.setenv("LLM_VALIDATOR_MAX_TOKENS", "600")
    monkeypatch.setenv("LLM_PACKET_MAX_CHARS", "2000")
    monkeypatch.setenv("LLM_APPENDIX_MAX_CHARS", "1200")
    monkeypatch.setenv("LLM_MAP_MAX_PACKETS", "2")
    monkeypatch.setenv("LLM_VALIDATION_MAX_ROUNDS", "1")
    monkeypatch.setenv("USE_LLM_ANALYSIS", "false")
    monkeypatch.setenv("USE_LLM_VALIDATOR", "false")
    monkeypatch.setenv("EVAL_GATE_ENABLED", "false")
    monkeypatch.setenv("EVAL_GATE_MIN_SCORE", "75")
    monkeypatch.setenv("EVAL_GATE_BLOCK_DRIVE_UPLOAD", "false")

    base = AgentConfig.from_env()
    profiled = _apply_weekly_quality_max_profile(base)

    assert profiled.gaia_model == QUALITY_MAX_GAIA_MODEL
    assert profiled.use_llm_analysis is True
    assert profiled.use_llm_validator is True
    assert profiled.eval_gate_enabled is True
    assert profiled.eval_gate_min_score >= 88
    assert profiled.eval_gate_block_drive_upload is True
    assert profiled.llm_map_max_tokens >= 900
    assert profiled.llm_reduce_max_tokens >= 2200
    assert profiled.llm_validator_max_tokens >= 1200
    assert profiled.llm_packet_max_chars >= 5200
    assert profiled.llm_appendix_max_chars >= 2600
    assert profiled.llm_map_max_packets >= 6
    assert profiled.llm_validation_max_rounds >= 3
