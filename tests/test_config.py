from weekly_seo_agent.config import AgentConfig


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
