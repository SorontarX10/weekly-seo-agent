# Sources and Integrations Documentation

## Scope

Ten dokument opisuje:

1. Jakie źródła danych są używane.
2. Jak są zaimplementowane integracje (klient, auth, env, retry/cache).
3. Jakie biblioteki i frameworki są używane.
4. Jak dodać nowe API w tym projekcie.

Repo root: `weekly_seo_agent/`

---

## 1. LLM + Orkiestracja

### GAIA / Azure OpenAI (LangChain)

- Cel: generacja `Executive summary`, `What is happening and why`, walidacja wieloetapowa.
- Implementacja:
  - `weekly_seo_agent/llm.py`
  - `weekly_seo_agent/workflow.py`
- Kluczowe zmienne:
  - `GAIA_ENDPOINT`
  - `GAIA_API_VERSION`
  - `GAIA_MODEL`
  - `GAIA_API_KEY` (lub `OPENAI_API_KEY`)
  - `GAIA_TIMEOUT_SEC`
  - `GAIA_MAX_RETRIES`
  - `GAIA_MAX_OUTPUT_TOKENS`
- Biblioteki:
  - `langchain-openai` (`AzureChatOpenAI`)
  - `langgraph`

Linki:

- https://python.langchain.com/docs/integrations/chat/azure_chat_openai/
- https://langchain-ai.github.io/langgraph/
- https://learn.microsoft.com/azure/ai-services/openai/

---

## 2. Core SEO Data Sources

### Google Search Console API

- Cel: baseline SEO (clicks, impressions, CTR, position), query/page movers, segmenty.
- Implementacja:
  - `weekly_seo_agent/clients/gsc_client.py`
  - token helper: `weekly_seo_agent/tools/gsc_refresh_token.py`
- Auth:
  - OAuth (`client_secret + refresh token`) lub JSON credentials path.
- Kluczowe zmienne:
  - `GSC_SITE_URL`
  - `GSC_OAUTH_CLIENT_SECRET_PATH`
  - `GSC_OAUTH_REFRESH_TOKEN`
  - opcjonalnie `GSC_CREDENTIALS_PATH`
  - `GSC_COUNTRY_FILTER`

Linki:

- https://developers.google.com/webmaster-tools/v1/searchanalytics/query
- https://developers.google.com/webmaster-tools/v1/quickstart/quickstart-python

### Senuto API

- Cel: visibility, content gap, competitor radar, SERP intelligence, movers.
- Implementacja:
  - `weekly_seo_agent/clients/senuto_client.py`
  - token helper: `weekly_seo_agent/tools/senuto_token.py`
- Auth:
  - Bearer token lub login/password (token refresh przez endpoint auth).
- Kluczowe zmienne:
  - `SENUTO_BASE_URL`
  - `SENUTO_TOKEN` lub `SENUTO_EMAIL` + `SENUTO_PASSWORD`
  - `SENUTO_COUNTRY_ID` / `SENUTO_COUNTRY_ID_MAP`
  - `SENUTO_TOP_ROWS`

Link:

- https://docs-api.senuto.com/

### Allegro Trends API

- Cel: dodatkowy sygnał demand z wyszukiwarki Allegro (VISIT/PV/OFFERS/GMV/DEALS).
- Implementacja:
  - `weekly_seo_agent/clients/allegro_trends_client.py`
- Auth:
  - OAuth password grant + basic auth (wewnętrzny endpoint Allegro Trends).
- Kluczowe zmienne:
  - `ALLEGRO_TRENDS_ENABLED`
  - `ALLEGRO_TRENDS_BASIC_AUTH_LOGIN`
  - `ALLEGRO_TRENDS_BASIC_AUTH_PASSWORD`
  - `ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_LOGIN`
  - `ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_PASSWORD`

Uwaga:

- W raporcie obowiązuje filtr szumu/brand-literówek, żeby nie wyciągać mylnych wniosków.

---

## 3. Analytics and Experience Context

### GA4 Data API

- Cel: kanały i kontekst ruchu (używane kontrolowanie polityką raportu).
- Implementacja:
  - `weekly_seo_agent/clients/ga4_client.py`
- Auth:
  - Service account JSON.
- Kluczowe zmienne:
  - `GA4_ENABLED`
  - `GA4_CREDENTIALS_PATH`
  - `GA4_PROPERTY_ID`
  - `GA4_PROPERTY_ID_MAP`

Link:

- https://developers.google.com/analytics/devguides/reporting/data/v1

### CrUX + PageSpeed Insights

- Cel: core web vitals jako kontekst (LCP/INP/CLS).
- Implementacja:
  - `weekly_seo_agent/additional_context.py`
- Auth:
  - API key (`PAGESPEED_API_KEY`) dla stabilności limitów.
- Kluczowe zmienne:
  - `PAGESPEED_API_KEY`
  - `TARGET_SITE_URL`

Linki:

- https://developer.chrome.com/docs/crux/api/
- https://developers.google.com/speed/docs/insights/v5/get-started

### Google Trends

- Cel:
  - trending topics,
  - brand-demand context (WoW/YoY) dla interpretacji brand/home.
- Implementacja:
  - `weekly_seo_agent/additional_context.py`
- Technika:
  - endpointy Trends (`explore`, `widgetdata/multiline`),
  - warmup session + cookie,
  - retry/backoff,
  - lokalny cache TTL.

Uwaga:

- To jest integracja oparta o nieoficjalny endpoint Trends web API (podatny na rate-limit/format changes).

---

## 4. External Market Signals

### Weather (history + forecast)

- Cel: rozróżnienie efektów popytowych (sezonowość) vs SEO.
- Implementacja:
  - `weekly_seo_agent/clients/external_signals.py`
- Źródło:
  - Open-Meteo.
- Kluczowe zmienne:
  - `WEATHER_LATITUDE` / `WEATHER_LONGITUDE`
  - mapy per kraj: `*_MAP`

Link:

- https://open-meteo.com/

### Holidays and calendar effects

- Cel: święta/ferie i ich wpływ na demand windows.
- Implementacja:
  - `weekly_seo_agent/clients/external_signals.py`
  - `weekly_seo_agent/ferie.py`
- Źródła:
  - OpenHolidays API + oficjalne ferie (agregowane z wiarygodnych źródeł publicznych).

Link:

- https://www.openholidaysapi.org/

### Google Search status + Search Central Blog

- Cel: kontekst update’ów i incydentów Google Search.
- Implementacja:
  - `weekly_seo_agent/clients/external_signals.py`
- Źródła:
  - status dashboard / blog RSS.

Linki:

- https://status.search.google.com/
- https://developers.google.com/search/blog

### News / market feeds

- Cel: wykrywanie wydarzeń rynkowych wpływających na GMV i popyt.
- Implementacja:
  - `weekly_seo_agent/clients/external_signals.py`
- Źródła:
  - RSS + HTML parsing konfigurowanych domen.

### GDELT (market-event API)

- Cel: event calendar i klasyfikacja potencjalnych shocków popytowych.
- Implementacja:
  - `weekly_seo_agent/additional_context.py`

Link:

- https://www.gdeltproject.org/

### DuckDuckGo Instant Answer API

- Cel: szybki zewnętrzny scouting hipotez (cause-oriented).
- Implementacja:
  - `weekly_seo_agent/additional_context.py`

Link:

- https://duckduckgo.com/api

### PL macro add-ons: NBP + IMGW

- Cel: kontekst walutowy i ostrzeżenia pogodowe dla rynku PL.
- Implementacja:
  - `weekly_seo_agent/additional_context.py`
- Źródła:
  - NBP API
  - IMGW warnings API

Linki:

- https://api.nbp.pl/
- https://danepubliczne.imgw.pl/api/data/warningsmeteo

---

## 5. Collaboration and Knowledge Inputs

### Google Drive + Google Docs API

- Cel:
  - publikacja raportu jako Google Doc,
  - odczyt prezentacji SEO i poprzednich raportów.
- Implementacja:
  - `weekly_seo_agent/clients/google_drive_client.py`
  - `weekly_seo_agent/clients/continuity_client.py`
- Auth:
  - OAuth token cache (`.google_drive_token.json`) lub service account.
- Kluczowe zmienne:
  - `GOOGLE_DRIVE_CREDENTIALS_PATH`
  - `GOOGLE_DRIVE_TOKEN_PATH`
  - `GOOGLE_DRIVE_FOLDER_ID`
  - `GOOGLE_DRIVE_REPORTS_FOLDER_NAME`

Linki:

- https://developers.google.com/drive/api
- https://developers.google.com/docs/api

### Google Sheets (status log, trend sheets)

- Cel:
  - continuity checks,
  - status topics,
  - current + next 31d trend inputs.
- Implementacja:
  - `weekly_seo_agent/clients/continuity_client.py`

Link:

- https://developers.google.com/sheets/api

---

## 6. Cross-Country model (PL/CZ/SK/HU)

Konfiguracja per kraj jest budowana w:

- `weekly_seo_agent/main.py`

Mapowane pola:

1. `gsc_country_filter`
2. `senuto_country_id`
3. `ga4_property_id`
4. weather coordinates/labels
5. country code dla external signals i Trends

Raporty uruchamiane są batchowo równolegle (oddzielne procesy).

---

## 7. Libraries Used

Z `pyproject.toml`:

1. `langgraph` - orkiestracja pipeline.
2. `langchain-openai` - LLM client (`AzureChatOpenAI`).
3. `google-api-python-client` - Drive/Docs/Sheets.
4. `google-auth`, `google-auth-oauthlib` - auth flows.
5. `requests` - HTTP clients.
6. `beautifulsoup4` - HTML parsing.
7. `python-dotenv` - env loading.
8. `python-docx` - DOCX rendering.
9. `pytest` (dev) - testy.

---

## 8. Integration Pattern Used in This Project

Każda nowa integracja powinna iść tym samym wzorcem:

1. `clients/<new_client>.py`
2. Konfiguracja w `config.py` + `.env.example`
3. Wywołanie w `workflow.py` lub `additional_context.py`
4. Normalizacja danych do prostego słownika/listy
5. Retry + timeout + soft-fail (error list, nie hard crash)
6. Cache TTL dla niestabilnych źródeł
7. Report mapping (`reporting.py`) z jasnym oznaczeniem źródła
8. Testy jednostkowe dla parserów i edge-case’ów

Minimalny kontrakt źródła:

1. `enabled: bool`
2. `source: str`
3. `rows: list[dict]`
4. `summary: dict`
5. `errors: list[str]`

---

## 9. Reliability and Failure Strategy

Aktualnie stosowane mechanizmy:

1. Retry/backoff (`429`, `5xx`) dla endpointów podatnych na limity.
2. Warmup session tam, gdzie endpoint wymaga cookie/handshake (Google Trends).
3. Fallback policy:
  - brak Trends brand => proxy na GSC brand queries,
  - brak trend-sheet YoY => fallback do GSC query trends.
4. Source TTL cache (redukcja flappingu i kosztów API).
5. Sekcje raportu oznaczają brak danych zamiast halucynować.

---

## 10. Quick Reference: Code Map

1. `weekly_seo_agent/main.py` - run batch, multiprocess, multi-country.
2. `weekly_seo_agent/workflow.py` - graph pipeline, prompt stages, validator loop.
3. `weekly_seo_agent/reporting.py` - narrative, summary, appendix, doc structure.
4. `weekly_seo_agent/additional_context.py` - market/news/weather/trends/macro.
5. `weekly_seo_agent/clients/gsc_client.py` - GSC.
6. `weekly_seo_agent/clients/senuto_client.py` - Senuto.
7. `weekly_seo_agent/clients/ga4_client.py` - GA4.
8. `weekly_seo_agent/clients/allegro_trends_client.py` - Allegro Trends.
9. `weekly_seo_agent/clients/google_drive_client.py` - Drive/Docs publish.
10. `weekly_seo_agent/clients/continuity_client.py` - previous reports, status, trend sheets.

