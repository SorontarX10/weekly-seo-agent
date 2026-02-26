from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _env(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default.strip()
    value = raw.strip()
    placeholder = f"{name}="
    unquoted = value.strip("'\"").strip()
    if unquoted.lower() == placeholder.lower():
        return default.strip()
    return value if value else default.strip()


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = _env(name, default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(values)


def _env_map_int(name: str, default: str = "") -> dict[str, int]:
    raw = _env(name, default)
    out: dict[str, int] = {}
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk or ":" not in chunk:
            continue
        key_raw, value_raw = chunk.split(":", 1)
        key = key_raw.strip().upper()
        value_text = value_raw.strip()
        if not key or not value_text:
            continue
        try:
            out[key] = int(value_text)
        except ValueError:
            continue
    return out


def _env_map_float(name: str, default: str = "") -> dict[str, float]:
    raw = _env(name, default)
    out: dict[str, float] = {}
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk or ":" not in chunk:
            continue
        key_raw, value_raw = chunk.split(":", 1)
        key = key_raw.strip().upper()
        value_text = value_raw.strip()
        if not key or not value_text:
            continue
        try:
            out[key] = float(value_text)
        except ValueError:
            continue
    return out


def _env_map_str(name: str, default: str = "") -> dict[str, str]:
    raw = _env(name, default)
    out: dict[str, str] = {}
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk or ":" not in chunk:
            continue
        key_raw, value_raw = chunk.split(":", 1)
        key = key_raw.strip().upper()
        value = value_raw.strip()
        if not key or not value:
            continue
        out[key] = value
    return out


def _normalize_target_site_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        value = "https://allegro.pl"
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"

    parsed = urlparse(value)
    host = parsed.netloc or parsed.path
    scheme = parsed.scheme or "https"
    host = host.strip().lower()
    return f"{scheme}://{host}/"


def _extract_host_from_url(raw: str) -> str:
    parsed = urlparse(raw)
    return (parsed.netloc or parsed.path).strip().lower()


def _normalize_gsc_country_filter(raw: str) -> str:
    value = raw.strip().strip("'\"")
    if not value:
        return ""

    lowered = value.lower()
    if lowered in {"all", "none"}:
        return ""

    upper = value.upper()
    if upper == "PL":
        return "pol"
    if len(upper) == 3 and upper.isalpha():
        return upper.lower()
    return lowered


@dataclass(frozen=True)
class AgentConfig:
    timezone: str
    output_dir: str
    top_n: int
    report_country_code: str
    report_countries: tuple[str, ...]
    google_drive_enabled: bool
    google_drive_client_secret_path: str
    google_drive_token_path: str
    google_drive_folder_name: str
    google_drive_folder_id: str
    seo_presentations_enabled: bool
    seo_presentations_folder_reference: str
    seo_presentations_max_files_per_year: int
    seo_presentations_max_text_files_per_year: int
    historical_reports_enabled: bool
    historical_reports_count: int
    historical_reports_yoy_tolerance_days: int
    status_log_enabled: bool
    status_file_reference: str
    status_max_rows: int
    product_trends_enabled: bool
    product_trends_comparison_sheet_reference: str
    product_trends_upcoming_sheet_reference: str
    product_trends_current_sheet_reference: str
    product_trends_top_rows: int
    product_trends_horizon_days: int
    merchant_center_enabled: bool
    merchant_center_credentials_path: str
    merchant_center_mca_id: str
    merchant_center_account_id_map: dict[str, str]
    trade_plan_enabled: bool
    trade_plan_sheet_reference: str
    trade_plan_yoy_sheet_reference: str
    trade_plan_tab_map: dict[str, str]
    trade_plan_yoy_tab_map: dict[str, str]
    trade_plan_top_rows: int
    platform_pulse_enabled: bool
    platform_pulse_rss_urls: tuple[str, ...]
    platform_pulse_top_rows: int
    governance_enabled: bool
    governance_human_reviewer: str
    eval_gate_enabled: bool
    eval_gate_min_score: int
    eval_gate_block_drive_upload: bool
    strict_llm_profile_enabled: bool
    startup_preflight_enabled: bool
    startup_preflight_blocking_sources: tuple[str, ...]
    ingestion_snapshot_enabled: bool
    ingestion_snapshot_retention_days: int
    telemetry_enabled: bool
    security_source_allowlist_domains: tuple[str, ...]

    target_site_url: str
    target_domain: str

    gsc_site_url: str
    gsc_site_url_map: dict[str, str]
    gsc_credentials_path: str
    gsc_oauth_client_secret_path: str
    gsc_oauth_refresh_token: str
    gsc_oauth_token_uri: str
    gsc_country_filter: str
    gsc_row_limit: int
    gsc_dimension_sets: tuple[tuple[str, ...], ...]

    senuto_token: str
    senuto_email: str
    senuto_password: str
    senuto_token_endpoint: str
    senuto_base_url: str
    senuto_domain: str
    senuto_visibility_endpoint: str
    senuto_fetch_mode: str
    senuto_country_id: int
    senuto_country_id_map: dict[str, int]
    senuto_date_interval: str
    senuto_visibility_metric: str
    senuto_competitor_domains: tuple[str, ...]
    senuto_top_rows: int

    ga4_enabled: bool
    ga4_property_id: str
    ga4_property_id_map: dict[str, str]
    ga4_credentials_path: str
    ga4_top_rows: int
    allegro_trends_enabled: bool
    allegro_trends_basic_auth_login: str
    allegro_trends_basic_auth_password: str
    allegro_trends_technical_account_login: str
    allegro_trends_technical_account_password: str
    allegro_trends_oauth_url: str
    allegro_trends_api_base_url: str
    allegro_trends_interval: str
    allegro_trends_exact: bool
    allegro_trends_escape_query: bool
    allegro_trends_measures: tuple[str, ...]
    allegro_trends_top_rows: int

    weather_latitude: float
    weather_longitude: float
    weather_label: str
    weather_context_enabled: bool
    weather_latitude_map: dict[str, float]
    weather_longitude_map: dict[str, float]
    weather_label_map: dict[str, str]

    google_status_endpoint: str
    google_blog_rss: str
    holidays_country_code: str
    holidays_country_code_map: dict[str, str]
    holidays_api_base_url: str
    holidays_language_code: str
    holidays_language_code_map: dict[str, str]
    pagespeed_api_key: str
    google_trends_rss_url: str
    google_trends_rss_url_map: dict[str, str]
    market_events_enabled: bool
    market_events_api_base_url: str
    market_events_top_rows: int
    free_public_sources_enabled: bool
    free_public_sources_top_rows: int
    nager_holidays_country_code: str
    eia_api_key: str
    nbp_api_base_url: str
    imgw_warnings_url: str
    news_scraping_enabled: bool
    news_rss_urls_pl: tuple[str, ...]
    news_rss_urls_global: tuple[str, ...]
    news_html_urls_pl: tuple[str, ...]
    news_html_urls_global: tuple[str, ...]
    news_keywords: tuple[str, ...]
    news_max_signals: int

    weekly_news_summary_enabled: bool
    weekly_news_domains_seo: tuple[str, ...]
    weekly_news_domains_geo: tuple[str, ...]
    weekly_news_max_items: int
    weekly_news_rss_urls_seo: tuple[str, ...]
    weekly_news_rss_urls_geo: tuple[str, ...]
    weekly_news_keywords_seo: tuple[str, ...]
    weekly_news_keywords_geo: tuple[str, ...]

    gmail_enabled: bool
    gmail_auth_mode: str
    gmail_service_account_path: str
    gmail_delegate_user: str
    gmail_sender: str
    gmail_recipient: str
    gmail_oauth_client_secret_path: str
    gmail_oauth_refresh_token: str
    gmail_oauth_token_uri: str

    min_click_loss_absolute: int
    min_click_loss_pct: float

    filter_irrelevant_queries: bool
    query_exclude_patterns: tuple[str, ...]

    use_llm_analysis: bool
    gaia_endpoint: str
    gaia_api_key: str
    gaia_api_version: str
    gaia_model: str
    gaia_temperature: float
    gaia_timeout_sec: int
    gaia_max_retries: int
    gaia_max_output_tokens: int
    llm_map_max_tokens: int
    llm_reduce_max_tokens: int
    llm_validator_max_tokens: int
    llm_packet_max_chars: int
    llm_appendix_max_chars: int
    llm_map_max_packets: int
    llm_validation_max_rounds: int
    use_llm_validator: bool
    cache_ttl_external_signals_sec: int
    cache_ttl_additional_context_sec: int
    cache_ttl_stale_fallback_sec: int
    source_ttl_weather_sec: int
    source_ttl_news_sec: int
    source_ttl_market_events_sec: int

    @classmethod
    def from_env(cls) -> "AgentConfig":
        target_site_url = _normalize_target_site_url(
            _env("TARGET_DOMAIN", "https://allegro.pl")
        )
        target_domain = _extract_host_from_url(target_site_url)

        dimension_sets_raw = _env("GSC_DIMENSION_SETS", "query;page")
        dimension_sets: list[tuple[str, ...]] = []
        for chunk in dimension_sets_raw.split(";"):
            dims = tuple(part.strip() for part in chunk.split(",") if part.strip())
            if dims:
                dimension_sets.append(dims)
        if not dimension_sets:
            dimension_sets = [("query",), ("page",)]

        default_news_rss_pl = (
            "https://news.google.com/rss/search?q=allegro+OR+e-commerce+OR+handel+OR+logistyka&hl=pl&gl=PL&ceid=PL:pl,"
            "https://www.bankier.pl/rss/wiadomosci.xml"
        )
        default_news_rss_global = (
            "https://news.google.com/rss/search?q=ecommerce+OR+retail+OR+google+search+update+OR+logistics&hl=en-US&gl=US&ceid=US:en,"
            "https://feeds.reuters.com/reuters/businessNews"
        )
        default_news_html_pl = "https://www.money.pl/,https://www.bankier.pl/"
        default_news_html_global = "https://www.reuters.com/world/,https://www.ft.com/world"
        default_news_keywords = (
            "allegro,e-commerce,ecommerce,retail,marketplace,sprzedaz,handel,logistyka,kurier,"
            "clo,vat,inflacja,stopy procentowe,google,core update,search update,awaria,outage,"
            "igrzyska,olimpiada,olympics,walentynki,wielkanoc"
        )
        default_report_countries = "PL,CZ,SK,HU"
        default_senuto_country_map = "PL:1,CZ:50,SK:164,HU:82"
        default_senuto_competitors = (
            "temu.com/pl,amazon.pl,ceneo.pl,olx.pl,shein.com,mediaexpert.pl"
        )
        default_weather_lat_map = "PL:52.2297,CZ:50.0755,SK:48.1486,HU:47.4979"
        default_weather_lon_map = "PL:21.0122,CZ:14.4378,SK:17.1077,HU:19.0402"
        default_weather_label_map = "PL:PL-Warsaw,CZ:CZ-Prague,SK:SK-Bratislava,HU:HU-Budapest"
        default_holidays_country_code_map = "PL:PL,CZ:CZ,SK:SK,HU:HU"
        default_holidays_language_code_map = "PL:PL,CZ:CS,SK:SK,HU:HU"
        default_google_trends_rss_map = (
            "PL:https://trends.google.com/trending/rss?geo=PL,"
            "CZ:https://trends.google.com/trending/rss?geo=CZ,"
            "SK:https://trends.google.com/trending/rss?geo=SK,"
            "HU:https://trends.google.com/trending/rss?geo=HU"
        )
        default_trade_plan_tab_map = (
            "PL:TP_2026_PL,CZ:TP_2026_CZ,SK:TP_2026_SK,HU:TP_2026_HU"
        )
        default_platform_pulse_rss = (
            "https://ec.europa.eu/commission/presscorner/api/rss,"
            "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=oj:JOL_2026_001_RSS,"
            "https://www.gov.pl/rss"
        )
        default_source_allowlist = (
            "google.com,googleapis.com,developers.google.com,searchengineland.com,searchenginejournal.com,"
            "seroundtable.com,reuters.com,ec.europa.eu,eur-lex.europa.eu,worldbank.org,"
            "trends.google.com,open-meteo.com,api.open-meteo.com,archive-api.open-meteo.com,"
            "gdeltproject.org,openholidaysapi.org,nager.at,api.nbp.pl,danepubliczne.imgw.pl,"
            "docs.google.com,drive.google.com,sheets.googleapis.com"
        )

        default_query_excludes = (
            "allegro ma non troppo,allegro non troppo,allegro con brio,allegro moderato,"
            "sonata allegro,tempo allegro,hotel allegro,restauracja allegro,kino allegro,"
            "allegro music,allegro muzyka"
        )

        default_news_domains_seo = (
            "searchengineland.com,searchenginejournal.com,seroundtable.com,developers.google.com/search,"
            "blog.google/products/search,blog.google/products"
        )
        default_news_domains_geo = (
            "openai.com,ai.googleblog.com,blog.google/technology/ai,anthropic.com,deepmind.google,"
            "cohere.com,perplexity.ai,openai.com/blog,ai.meta.com"
        )
        default_weekly_news_rss_seo = (
            "https://news.google.com/rss/search?q=google+search+update+OR+seo+OR+algorithm+update&hl=en-US&gl=US&ceid=US:en,"
            "https://www.searchenginejournal.com/feed/,https://searchengineland.com/feed,https://www.seroundtable.com/index.xml,"
            "https://developers.google.com/search/blog/rss.xml"
        )
        default_weekly_news_rss_geo = (
            "https://news.google.com/rss/search?q=generative+search+OR+ai+search+OR+answer+engine+OR+SGE&hl=en-US&gl=US&ceid=US:en,"
            "https://openai.com/blog/rss.xml,https://ai.googleblog.com/feeds/posts/default?alt=rss,"
            "https://deepmind.google/discover/rss.xml,https://www.anthropic.com/news/rss.xml,"
            "https://www.perplexity.ai/hub/rss"
        )
        default_weekly_news_keywords_seo = (
            "seo,search,google,algorithm,core update,ranking,indexing,search console,discover"
        )
        default_weekly_news_keywords_geo = (
            "generative,ai search,answer engine,sge,search generative experience,overview,llm,chat"
        )

        gsc_site_default = target_site_url
        senuto_domain_default = target_domain

        return cls(
            timezone=_env("SCHEDULE_TZ", "Europe/Warsaw"),
            output_dir=_env("OUTPUT_DIR", "SEO Weekly Reports"),
            top_n=_env_int("TOP_N", 25),
            report_country_code=_env("REPORT_COUNTRY_CODE", "PL").upper(),
            report_countries=tuple(
                country.upper()
                for country in _env_csv("REPORT_COUNTRIES", default_report_countries)
            ),
            google_drive_enabled=_env_bool("GOOGLE_DRIVE_ENABLED", True),
            google_drive_client_secret_path=_env(
                "GOOGLE_DRIVE_CLIENT_SECRET_PATH", "secret.json"
            ),
            google_drive_token_path=_env(
                "GOOGLE_DRIVE_TOKEN_PATH", ".google_drive_token.json"
            ),
            google_drive_folder_name=_env(
                "GOOGLE_DRIVE_FOLDER_NAME", "SEO Weekly Reports"
            ),
            google_drive_folder_id=_env("GOOGLE_DRIVE_FOLDER_ID"),
            seo_presentations_enabled=_env_bool("SEO_PRESENTATIONS_ENABLED", False),
            seo_presentations_folder_reference=_env("SEO_PRESENTATIONS_FOLDER_REFERENCE"),
            seo_presentations_max_files_per_year=_env_int(
                "SEO_PRESENTATIONS_MAX_FILES_PER_YEAR", 20
            ),
            seo_presentations_max_text_files_per_year=_env_int(
                "SEO_PRESENTATIONS_MAX_TEXT_FILES_PER_YEAR", 8
            ),
            historical_reports_enabled=_env_bool("HISTORICAL_REPORTS_ENABLED", True),
            historical_reports_count=_env_int("HISTORICAL_REPORTS_COUNT", 3),
            historical_reports_yoy_tolerance_days=_env_int(
                "HISTORICAL_REPORTS_YOY_TOLERANCE_DAYS", 28
            ),
            status_log_enabled=_env_bool("STATUS_LOG_ENABLED", True),
            status_file_reference=_env("STATUS_FILE_REFERENCE"),
            status_max_rows=_env_int("STATUS_MAX_ROWS", 12),
            product_trends_enabled=_env_bool("PRODUCT_TRENDS_ENABLED", True),
            product_trends_comparison_sheet_reference=_env(
                "PRODUCT_TRENDS_COMPARISON_SHEET_REFERENCE"
            ),
            product_trends_upcoming_sheet_reference=_env(
                "PRODUCT_TRENDS_UPCOMING_SHEET_REFERENCE"
            ),
            product_trends_current_sheet_reference=_env(
                "PRODUCT_TRENDS_CURRENT_SHEET_REFERENCE"
            ),
            product_trends_top_rows=_env_int("PRODUCT_TRENDS_TOP_ROWS", 12),
            product_trends_horizon_days=_env_int("PRODUCT_TRENDS_HORIZON_DAYS", 31),
            merchant_center_enabled=_env_bool("MERCHANT_CENTER_ENABLED", False),
            merchant_center_credentials_path=_env("MERCHANT_CENTER_CREDENTIALS_PATH"),
            merchant_center_mca_id=_env("MERCHANT_CENTER_MCA_ID"),
            merchant_center_account_id_map=_env_map_str(
                "MERCHANT_CENTER_ACCOUNT_ID_MAP"
            ),
            trade_plan_enabled=_env_bool("TRADE_PLAN_ENABLED", True),
            trade_plan_sheet_reference=_env("TRADE_PLAN_SHEET_REFERENCE"),
            trade_plan_yoy_sheet_reference=_env("TRADE_PLAN_YOY_SHEET_REFERENCE"),
            trade_plan_tab_map=_env_map_str("TRADE_PLAN_TAB_MAP", default_trade_plan_tab_map),
            trade_plan_yoy_tab_map=_env_map_str("TRADE_PLAN_YOY_TAB_MAP"),
            trade_plan_top_rows=_env_int("TRADE_PLAN_TOP_ROWS", 12),
            platform_pulse_enabled=_env_bool("PLATFORM_PULSE_ENABLED", True),
            platform_pulse_rss_urls=_env_csv(
                "PLATFORM_PULSE_RSS_URLS", default_platform_pulse_rss
            ),
            platform_pulse_top_rows=_env_int("PLATFORM_PULSE_TOP_ROWS", 12),
            governance_enabled=_env_bool("GOVERNANCE_ENABLED", True),
            governance_human_reviewer=_env("GOVERNANCE_HUMAN_REVIEWER"),
            eval_gate_enabled=_env_bool("EVAL_GATE_ENABLED", True),
            eval_gate_min_score=_env_int("EVAL_GATE_MIN_SCORE", 75),
            eval_gate_block_drive_upload=_env_bool("EVAL_GATE_BLOCK_DRIVE_UPLOAD", True),
            strict_llm_profile_enabled=_env_bool(
                "WEEKLY_STRICT_LLM_PROFILE_ENABLED", True
            ),
            startup_preflight_enabled=_env_bool("STARTUP_PREFLIGHT_ENABLED", True),
            startup_preflight_blocking_sources=tuple(
                token.strip().lower()
                for token in _env_csv(
                    "STARTUP_PREFLIGHT_BLOCKING_SOURCES",
                    "gsc,drive",
                )
                if token.strip()
            ),
            ingestion_snapshot_enabled=_env_bool("INGESTION_SNAPSHOT_ENABLED", True),
            ingestion_snapshot_retention_days=max(
                1, _env_int("INGESTION_SNAPSHOT_RETENTION_DAYS", 45)
            ),
            telemetry_enabled=_env_bool("TELEMETRY_ENABLED", True),
            security_source_allowlist_domains=_env_csv(
                "SECURITY_SOURCE_ALLOWLIST_DOMAINS", default_source_allowlist
            ),
            target_site_url=target_site_url,
            target_domain=target_domain,
            gsc_site_url=_env("GSC_SITE_URL", gsc_site_default),
            gsc_site_url_map=_env_map_str("GSC_SITE_URL_MAP"),
            gsc_credentials_path=_env("GSC_CREDENTIALS_PATH"),
            gsc_oauth_client_secret_path=_env("GSC_OAUTH_CLIENT_SECRET_PATH"),
            gsc_oauth_refresh_token=_env("GSC_OAUTH_REFRESH_TOKEN"),
            gsc_oauth_token_uri=_env("GSC_OAUTH_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            gsc_country_filter=_normalize_gsc_country_filter(_env("GSC_COUNTRY_FILTER", "PL")),
            gsc_row_limit=_env_int("GSC_ROW_LIMIT", 3000),
            gsc_dimension_sets=tuple(dimension_sets),
            senuto_token=_env("SENUTO_TOKEN"),
            senuto_email=_env("SENUTO_EMAIL"),
            senuto_password=_env("SENUTO_PASSWORD"),
            senuto_token_endpoint=_env("SENUTO_TOKEN_ENDPOINT", "/api/users/token"),
            senuto_base_url=_env("SENUTO_BASE_URL", "https://api.senuto.com"),
            senuto_domain=_env("SENUTO_DOMAIN", senuto_domain_default),
            senuto_visibility_endpoint=_env(
                "SENUTO_VISIBILITY_ENDPOINT",
                "/api/visibility_analysis/reports/domain_positions/getPositionsSumsDistributionChartData",
            ),
            senuto_fetch_mode=_env("SENUTO_FETCH_MODE", "topLevelDomain"),
            senuto_country_id=_env_int("SENUTO_COUNTRY_ID", 1),
            senuto_country_id_map=_env_map_int(
                "SENUTO_COUNTRY_ID_MAP", default_senuto_country_map
            ),
            senuto_date_interval=_env("SENUTO_DATE_INTERVAL", "weekly"),
            senuto_visibility_metric=_env("SENUTO_VISIBILITY_METRIC", "top10"),
            senuto_competitor_domains=_env_csv(
                "SENUTO_COMPETITOR_DOMAINS",
                default_senuto_competitors,
            ),
            senuto_top_rows=_env_int("SENUTO_TOP_ROWS", 10),
            ga4_enabled=_env_bool("GA4_ENABLED", False),
            ga4_property_id=_env("GA4_PROPERTY_ID"),
            ga4_property_id_map=_env_map_str("GA4_PROPERTY_ID_MAP"),
            ga4_credentials_path=_env("GA4_CREDENTIALS_PATH", "secret.json"),
            ga4_top_rows=_env_int("GA4_TOP_ROWS", 10),
            allegro_trends_enabled=_env_bool("ALLEGRO_TRENDS_ENABLED", False),
            allegro_trends_basic_auth_login=_env(
                "ALLEGRO_TRENDS_BASIC_AUTH_LOGIN", "search-trends-ui"
            ),
            allegro_trends_basic_auth_password=_env(
                "ALLEGRO_TRENDS_BASIC_AUTH_PASSWORD"
            ),
            allegro_trends_technical_account_login=_env(
                "ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_LOGIN"
            ),
            allegro_trends_technical_account_password=_env(
                "ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_PASSWORD"
            ),
            allegro_trends_oauth_url=_env(
                "ALLEGRO_TRENDS_OAUTH_URL",
                "https://oauth.allegrogroup.com/auth/oauth/token",
            ),
            allegro_trends_api_base_url=_env(
                "ALLEGRO_TRENDS_API_BASE_URL",
                "https://search-trends-service-prod-passive.allegrogroup.com",
            ),
            allegro_trends_interval=_env("ALLEGRO_TRENDS_INTERVAL", "day"),
            allegro_trends_exact=_env_bool("ALLEGRO_TRENDS_EXACT", False),
            allegro_trends_escape_query=_env_bool(
                "ALLEGRO_TRENDS_ESCAPE_QUERY", True
            ),
            allegro_trends_measures=tuple(
                part.strip().upper()
                for part in _env_csv(
                    "ALLEGRO_TRENDS_MEASURES", "VISIT,PV,OFFERS,GMV,DEALS"
                )
                if part.strip()
            ),
            allegro_trends_top_rows=_env_int("ALLEGRO_TRENDS_TOP_ROWS", 10),
            weather_latitude=_env_float("WEATHER_LATITUDE", 52.2297),
            weather_longitude=_env_float("WEATHER_LONGITUDE", 21.0122),
            weather_label=_env("WEATHER_LABEL", "PL-central"),
            weather_context_enabled=_env_bool("WEATHER_CONTEXT_ENABLED", True),
            weather_latitude_map=_env_map_float(
                "WEATHER_LATITUDE_MAP", default_weather_lat_map
            ),
            weather_longitude_map=_env_map_float(
                "WEATHER_LONGITUDE_MAP", default_weather_lon_map
            ),
            weather_label_map=_env_map_str(
                "WEATHER_LABEL_MAP", default_weather_label_map
            ),
            google_status_endpoint=_env(
                "GOOGLE_STATUS_ENDPOINT",
                "https://status.search.google.com/incidents.json",
            ),
            google_blog_rss=_env(
                "GOOGLE_BLOG_RSS",
                "https://feeds.feedburner.com/blogspot/amDG",
            ),
            holidays_country_code=_env("HOLIDAYS_COUNTRY_CODE", "PL"),
            holidays_country_code_map=_env_map_str(
                "HOLIDAYS_COUNTRY_CODE_MAP",
                default_holidays_country_code_map,
            ),
            holidays_api_base_url=_env(
                "HOLIDAYS_API_BASE_URL",
                "https://openholidaysapi.org",
            ),
            holidays_language_code=_env("HOLIDAYS_LANGUAGE_CODE", "PL"),
            holidays_language_code_map=_env_map_str(
                "HOLIDAYS_LANGUAGE_CODE_MAP",
                default_holidays_language_code_map,
            ),
            pagespeed_api_key=_env("PAGESPEED_API_KEY"),
            google_trends_rss_url=_env(
                "GOOGLE_TRENDS_RSS_URL",
                "https://trends.google.com/trending/rss?geo=PL",
            ),
            google_trends_rss_url_map=_env_map_str(
                "GOOGLE_TRENDS_RSS_URL_MAP", default_google_trends_rss_map
            ),
            market_events_enabled=_env_bool("MARKET_EVENTS_ENABLED", True),
            market_events_api_base_url=_env(
                "MARKET_EVENTS_API_BASE_URL",
                "https://api.gdeltproject.org/api/v2/doc/doc",
            ),
            market_events_top_rows=_env_int("MARKET_EVENTS_TOP_ROWS", 12),
            free_public_sources_enabled=_env_bool("FREE_PUBLIC_SOURCES_ENABLED", True),
            free_public_sources_top_rows=_env_int("FREE_PUBLIC_SOURCES_TOP_ROWS", 3),
            nager_holidays_country_code=_env("NAGER_HOLIDAYS_COUNTRY_CODE", _env("HOLIDAYS_COUNTRY_CODE", "PL")).upper(),
            eia_api_key=_env("EIA_API_KEY"),
            nbp_api_base_url=_env("NBP_API_BASE_URL", "https://api.nbp.pl/api"),
            imgw_warnings_url=_env(
                "IMGW_WARNINGS_URL",
                "https://danepubliczne.imgw.pl/api/data/warningsmeteo",
            ),
            news_scraping_enabled=_env_bool("NEWS_SCRAPING_ENABLED", True),
            news_rss_urls_pl=_env_csv("NEWS_RSS_URLS_PL", default_news_rss_pl),
            news_rss_urls_global=_env_csv("NEWS_RSS_URLS_GLOBAL", default_news_rss_global),
            news_html_urls_pl=_env_csv("NEWS_HTML_URLS_PL", default_news_html_pl),
            news_html_urls_global=_env_csv("NEWS_HTML_URLS_GLOBAL", default_news_html_global),
            news_keywords=_env_csv("NEWS_KEYWORDS", default_news_keywords),
            news_max_signals=_env_int("NEWS_MAX_SIGNALS", 20),

            weekly_news_summary_enabled=_env_bool("WEEKLY_NEWS_SUMMARY_ENABLED", True),
            weekly_news_domains_seo=_env_csv("WEEKLY_NEWS_DOMAINS_SEO", default_news_domains_seo),
            weekly_news_domains_geo=_env_csv("WEEKLY_NEWS_DOMAINS_GEO", default_news_domains_geo),
            weekly_news_max_items=_env_int("WEEKLY_NEWS_MAX_ITEMS", 12),
            weekly_news_rss_urls_seo=_env_csv(
                "WEEKLY_NEWS_RSS_URLS_SEO", default_weekly_news_rss_seo
            ),
            weekly_news_rss_urls_geo=_env_csv(
                "WEEKLY_NEWS_RSS_URLS_GEO", default_weekly_news_rss_geo
            ),
            weekly_news_keywords_seo=_env_csv(
                "WEEKLY_NEWS_KEYWORDS_SEO", default_weekly_news_keywords_seo
            ),
            weekly_news_keywords_geo=_env_csv(
                "WEEKLY_NEWS_KEYWORDS_GEO", default_weekly_news_keywords_geo
            ),

            gmail_enabled=_env_bool("GMAIL_ENABLED", False),
            gmail_auth_mode=_env("GMAIL_AUTH_MODE", "service_account").lower(),
            gmail_service_account_path=_env("GMAIL_SERVICE_ACCOUNT_PATH", "secret.json"),
            gmail_delegate_user=_env("GMAIL_DELEGATE_USER"),
            gmail_sender=_env("GMAIL_SENDER"),
            gmail_recipient=_env("GMAIL_RECIPIENT"),
            gmail_oauth_client_secret_path=_env("GMAIL_OAUTH_CLIENT_SECRET_PATH"),
            gmail_oauth_refresh_token=_env("GMAIL_OAUTH_REFRESH_TOKEN"),
            gmail_oauth_token_uri=_env(
                "GMAIL_OAUTH_TOKEN_URI", "https://oauth2.googleapis.com/token"
            ),
            min_click_loss_absolute=_env_int("MIN_CLICK_LOSS_ABSOLUTE", 100),
            min_click_loss_pct=_env_float("MIN_CLICK_LOSS_PCT", 0.15),
            filter_irrelevant_queries=_env_bool("FILTER_IRRELEVANT_QUERIES", True),
            query_exclude_patterns=_env_csv("QUERY_EXCLUDE_PATTERNS", default_query_excludes),
            use_llm_analysis=_env_bool("USE_LLM_ANALYSIS", True),
            gaia_endpoint=_env("GAIA_ENDPOINT", _env("AZURE_OPENAI_ENDPOINT")),
            gaia_api_key=_env(
                "GAIA_API_KEY",
                _env("OPENAI_API_KEY", _env("AZURE_OPENAI_API_KEY")),
            ),
            gaia_api_version=_env("GAIA_API_VERSION", _env("AZURE_OPENAI_API_VERSION")),
            gaia_model=_env("GAIA_MODEL", _env("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")),
            gaia_temperature=_env_float(
                "GAIA_TEMPERATURE",
                _env_float("AZURE_OPENAI_TEMPERATURE", 0.0),
            ),
            gaia_timeout_sec=_env_int("GAIA_TIMEOUT_SEC", 120),
            gaia_max_retries=_env_int("GAIA_MAX_RETRIES", 1),
            gaia_max_output_tokens=_env_int("GAIA_MAX_OUTPUT_TOKENS", 1400),
            llm_map_max_tokens=_env_int("LLM_MAP_MAX_TOKENS", 500),
            llm_reduce_max_tokens=_env_int("LLM_REDUCE_MAX_TOKENS", 1400),
            llm_validator_max_tokens=_env_int("LLM_VALIDATOR_MAX_TOKENS", 800),
            llm_packet_max_chars=_env_int("LLM_PACKET_MAX_CHARS", 3200),
            llm_appendix_max_chars=_env_int("LLM_APPENDIX_MAX_CHARS", 1800),
            llm_map_max_packets=_env_int("LLM_MAP_MAX_PACKETS", 4),
            llm_validation_max_rounds=_env_int("LLM_VALIDATION_MAX_ROUNDS", 2),
            use_llm_validator=_env_bool("USE_LLM_VALIDATOR", True),
            cache_ttl_external_signals_sec=_env_int("CACHE_TTL_EXTERNAL_SIGNALS_SEC", 6 * 3600),
            cache_ttl_additional_context_sec=_env_int("CACHE_TTL_ADDITIONAL_CONTEXT_SEC", 6 * 3600),
            cache_ttl_stale_fallback_sec=_env_int("CACHE_TTL_STALE_FALLBACK_SEC", 30 * 24 * 3600),
            source_ttl_weather_sec=_env_int("SOURCE_TTL_WEATHER_SEC", 24 * 3600),
            source_ttl_news_sec=_env_int("SOURCE_TTL_NEWS_SEC", 6 * 3600),
            source_ttl_market_events_sec=_env_int("SOURCE_TTL_MARKET_EVENTS_SEC", 48 * 3600),
        )

    @property
    def senuto_enabled(self) -> bool:
        has_token = bool(self.senuto_token)
        has_credentials = bool(self.senuto_email and self.senuto_password)
        return bool(
            (has_token or has_credentials)
            and self.senuto_domain
            and self.senuto_visibility_endpoint
        )

    @property
    def gsc_enabled(self) -> bool:
        if not self.gsc_site_url:
            return False
        has_service_account = bool(self.gsc_credentials_path)
        has_oauth = bool(
            self.gsc_oauth_client_secret_path and self.gsc_oauth_refresh_token
        )
        return has_service_account or has_oauth

    @property
    def gaia_llm_enabled(self) -> bool:
        return bool(
            self.gaia_endpoint
            and self.gaia_api_key
            and self.gaia_api_version
            and self.gaia_model
        )

    @property
    def merchant_center_api_enabled(self) -> bool:
        return bool(
            self.merchant_center_enabled
            and self.merchant_center_credentials_path
        )

    @property
    def google_drive_upload_enabled(self) -> bool:
        return bool(
            self.google_drive_enabled
            and self.google_drive_client_secret_path
            and self.google_drive_folder_name
        )

    @property
    def ga4_reporting_enabled(self) -> bool:
        return bool(
            self.ga4_enabled
            and self.ga4_property_id
            and self.ga4_credentials_path
        )

    @property
    def allegro_trends_api_enabled(self) -> bool:
        return bool(
            self.allegro_trends_enabled
            and self.allegro_trends_basic_auth_login
            and self.allegro_trends_basic_auth_password
            and self.allegro_trends_technical_account_login
            and self.allegro_trends_technical_account_password
            and self.allegro_trends_oauth_url
            and self.allegro_trends_api_base_url
        )
