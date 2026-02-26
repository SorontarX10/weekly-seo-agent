from weekly_seo_agent.config import AgentConfig
from weekly_seo_agent.weekly_reporting_agent.config import AgentConfig as WeeklyAgentConfig
from weekly_seo_agent.weekly_reporting_agent.main import (
    QUALITY_MAX_GAIA_MODEL,
    _apply_weekly_quality_max_profile,
    _apply_runtime_toggles,
    _is_gsc_mapping_explicit,
    _preflight_severity,
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


def test_weekly_agent_merchant_center_config_from_env(monkeypatch):
    monkeypatch.setenv("MERCHANT_CENTER_ENABLED", "true")
    monkeypatch.setenv("MERCHANT_CENTER_CREDENTIALS_PATH", "secret.json")
    monkeypatch.setenv("MERCHANT_CENTER_MCA_ID", "123456")
    monkeypatch.setenv("MERCHANT_CENTER_ACCOUNT_ID_MAP", "PL:111,CZ:222")
    cfg = WeeklyAgentConfig.from_env()
    assert cfg.merchant_center_enabled is True
    assert cfg.merchant_center_api_enabled is True
    assert cfg.merchant_center_credentials_path == "secret.json"
    assert cfg.merchant_center_mca_id == "123456"
    assert cfg.merchant_center_account_id_map["PL"] == "111"
    assert cfg.merchant_center_account_id_map["CZ"] == "222"


def test_runtime_toggles_can_disable_strict_profile(monkeypatch):
    monkeypatch.setenv("GAIA_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("WEEKLY_STRICT_LLM_PROFILE_ENABLED", "true")
    cfg = WeeklyAgentConfig.from_env()

    toggled = _apply_runtime_toggles(
        cfg,
        merchant_center_enabled=True,
        strict_llm_profile_enabled=False,
        enable_sources=[],
        disable_sources=[],
    )
    assert toggled.merchant_center_enabled is True
    assert toggled.strict_llm_profile_enabled is False
    assert toggled.gaia_model == "gpt-4o-mini"


def test_runtime_toggles_can_force_strict_profile(monkeypatch):
    monkeypatch.setenv("GAIA_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("WEEKLY_STRICT_LLM_PROFILE_ENABLED", "false")
    cfg = WeeklyAgentConfig.from_env()

    toggled = _apply_runtime_toggles(
        cfg,
        merchant_center_enabled=None,
        strict_llm_profile_enabled=True,
        enable_sources=[],
        disable_sources=[],
    )
    assert toggled.strict_llm_profile_enabled is True
    assert toggled.gaia_model == QUALITY_MAX_GAIA_MODEL


def test_runtime_source_toggles(monkeypatch):
    monkeypatch.setenv("NEWS_SCRAPING_ENABLED", "true")
    monkeypatch.setenv("WEATHER_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("TRADE_PLAN_ENABLED", "true")
    monkeypatch.setenv("WEEKLY_STRICT_LLM_PROFILE_ENABLED", "false")
    cfg = WeeklyAgentConfig.from_env()

    toggled = _apply_runtime_toggles(
        cfg,
        merchant_center_enabled=None,
        strict_llm_profile_enabled=False,
        enable_sources=["news"],
        disable_sources=["weather", "trade-plan"],
    )
    assert toggled.news_scraping_enabled is True
    assert toggled.weather_context_enabled is False
    assert toggled.trade_plan_enabled is False


def test_startup_preflight_defaults(monkeypatch):
    monkeypatch.delenv("STARTUP_PREFLIGHT_ENABLED", raising=False)
    monkeypatch.delenv("STARTUP_PREFLIGHT_BLOCKING_SOURCES", raising=False)
    cfg = WeeklyAgentConfig.from_env()
    assert cfg.startup_preflight_enabled is True
    assert "gsc" in cfg.startup_preflight_blocking_sources
    assert "drive" in cfg.startup_preflight_blocking_sources


def test_preflight_severity_uses_blocking_sources(monkeypatch):
    monkeypatch.setenv("STARTUP_PREFLIGHT_BLOCKING_SOURCES", "gsc,drive")
    cfg = WeeklyAgentConfig.from_env()
    assert _preflight_severity(cfg, "gsc") == "blocker"
    assert _preflight_severity(cfg, "sheets") == "warning"


def test_ingestion_snapshot_defaults(monkeypatch):
    monkeypatch.delenv("INGESTION_SNAPSHOT_ENABLED", raising=False)
    monkeypatch.delenv("INGESTION_SNAPSHOT_RETENTION_DAYS", raising=False)
    cfg = WeeklyAgentConfig.from_env()
    assert cfg.ingestion_snapshot_enabled is True
    assert cfg.ingestion_snapshot_retention_days == 45


def test_ingestion_snapshot_env_override(monkeypatch):
    monkeypatch.setenv("INGESTION_SNAPSHOT_ENABLED", "false")
    monkeypatch.setenv("INGESTION_SNAPSHOT_RETENTION_DAYS", "10")
    cfg = WeeklyAgentConfig.from_env()
    assert cfg.ingestion_snapshot_enabled is False
    assert cfg.ingestion_snapshot_retention_days == 10


def test_gsc_mapping_explicit_only_when_all_countries_present(monkeypatch):
    monkeypatch.setenv(
        "GSC_SITE_URL_MAP",
        "PL:https://allegro.pl,CZ:https://allegro.cz,SK:https://allegro.sk,HU:https://allegro.hu",
    )
    cfg = WeeklyAgentConfig.from_env()
    assert _is_gsc_mapping_explicit(cfg, ["PL", "CZ", "SK", "HU"]) is True
    assert _is_gsc_mapping_explicit(cfg, ["PL", "DE"]) is False
