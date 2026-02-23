# Weekly SEO Intelligence Agent

Tygodniowy agent SEO (LangGraph + GAIA), ktory analizuje:

Pelna dokumentacja architektury i odtworzenia projektu:
`docs/PROJECT_DOCUMENTATION.md`

Szybki onboarding tylko dla agenta raportowania tygodniowego (share-ready):
`weekly_seo_agent/weekly_reporting_agent/README.md`

1. GSC: `poprzedni pelny tydzien (pon-nd)`, `tydzien wczesniej`, `YoY = 52 tygodnie wstecz (te same dni tygodnia)`.
2. Senuto visibility.
3. Sygnaly zewnetrzne: pogoda, swieta/ferie (OpenHolidays API), status Google, scraping newsow per rynek + global.
4. Dodatkowe zrodla: CrUX/PageSpeed, Google Trends per rynek, NBP FX i IMGW warnings (dla PL).
5. Filtruje frazy niepowiazane z Allegro (np. muzyczne "allegro ...").
6. W okresie ferii pokazuje roznice regionalne YoY (wojewodztwa) i proxy sily GMV oparte o populacje i srednie zarobki.
7. Dodaje kontekst z prezentacji zespolu SEO (Google Drive: biezacy + poprzedni rok, foldery roczne).
8. Dodaje continuity context z poprzednich raportow (2-3 ostatnie + raport YoY dla tego samego okresu, jesli istnieje).
9. Dodaje kontekst z pliku statusowego (Google Sheets, zakladki roczne, kolumna dat statusu).
10. Raport zawiera Root Cause Matrix, confidence score + evidence i segmentacje GSC (brand/non-brand, device, page name).
11. Raport jest generowany po angielsku (AI commentary + data appendix).
12. Dodaje trendy produktowe non-brand z Google Sheets: top trendy YoY (ten rok vs poprzedni), top current trends i top upcoming 31 dni.
13. Dodaje analize akcji promocyjnych marketplace: Allegro (np. Black Week, Smart Week, Allegro Days, Megaraty) vs konkurencja.
14. Dodaje modul Senuto competitor intelligence: competitor radar, content gap, keyword movers, acquired/lost, direct answers, seasonality, market ranking, trending keywords, SERP volatility.
15. Dodaje wielorynkowe raportowanie per kraj (`PL`, `CZ`, `SK`, `HU`) w jednym uruchomieniu.
16. Generuje oddzielne raporty per kraj (`PL`, `CZ`, `SK`, `HU`) w jednym uruchomieniu.
17. Dodaje kalendarz wydarzen rynkowych per kraj z API (`GDELT DOC API`) i ocene potencjalnego wplywu na GMV.
18. Opcjonalnie dodaje sygnaly popytu z Allegro Trends API (VISIT/PV/OFFERS/GMV/DEALS) dla top mover queries.
19. Stabilizuje zrodla zewnetrzne: retry + backoff + timeout + fallback do ostatniego poprawnego cache (stale), z oznaczeniem degradacji w sygnalach.
20. Dodaje warstwe decyzyjna: `Decision one-pager`, `Impact attribution`, `Query anomaly detection`, `Data quality score`, `Hypothesis tracker`.

## Kluczowe zalozenia

- Domyslny target: `https://allegro.pl/` (URL-prefix, bez subdomen).
- LLM jest konfigurowany przez `GAIA_*`.
- Senuto moze dzialac na:
  - `SENUTO_TOKEN`, albo
  - `SENUTO_EMAIL + SENUTO_PASSWORD` (token pobierany automatycznie i odswiezany po `401`).

## Instalacja

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Co skonfigurowac, aby dzialalo

W `.env` musisz miec:

1. GAIA:
- `GAIA_ENDPOINT`
- `GAIA_API_VERSION`
- `GAIA_MODEL`
- `GAIA_API_KEY` (albo `OPENAI_API_KEY`)
- `GAIA_TIMEOUT_SEC=120` (zalecane, zeby pojedynczy kraj nie zawisal na LLM)
- `GAIA_MAX_RETRIES=1`
- `GAIA_MAX_OUTPUT_TOKENS=1400` (globalny cap output tokenow na call)
- `USE_LLM_VALIDATOR=true` (mozna wylaczyc, jesli chcesz maksymalnie oszczedny run)
- `LLM_MAP_MAX_TOKENS=500`, `LLM_REDUCE_MAX_TOKENS=1400`, `LLM_VALIDATOR_MAX_TOKENS=800`
- `LLM_PACKET_MAX_CHARS=3200`, `LLM_APPENDIX_MAX_CHARS=1800`, `LLM_MAP_MAX_PACKETS=4`
- `LLM_VALIDATION_MAX_ROUNDS=2`
- `CACHE_TTL_EXTERNAL_SIGNALS_SEC=21600`, `CACHE_TTL_ADDITIONAL_CONTEXT_SEC=21600`
- `CACHE_TTL_STALE_FALLBACK_SEC=2592000`
- `SOURCE_TTL_NEWS_SEC=21600`, `SOURCE_TTL_WEATHER_SEC=86400`, `SOURCE_TTL_MARKET_EVENTS_SEC=172800`

2. GSC:
- `GSC_OAUTH_CLIENT_SECRET_PATH`
- `GSC_OAUTH_REFRESH_TOKEN`
- `GSC_COUNTRY_FILTER=PL` (domyslnie filtrujemy tylko ruch z Polski; `none` wylacza filtr)

3. Senuto (jedna z opcji):
- `SENUTO_TOKEN`
- albo `SENUTO_EMAIL` + `SENUTO_PASSWORD`
- Konfiguracja rynkow: `SENUTO_COUNTRY_ID_MAP="PL:1,CZ:50,SK:164,HU:82"`
- Lista konkurentow: `SENUTO_COMPETITOR_DOMAINS="temu.com/pl,amazon.pl,ceneo.pl,olx.pl,shein.com,mediaexpert.pl"`
- Limit top rows dla sekcji Senuto: `SENUTO_TOP_ROWS=10`

4. Kraje raportowania:
- `REPORT_COUNTRIES="PL,CZ,SK,HU"`
- (opcjonalnie pojedynczy run) `REPORT_COUNTRY_CODE=PL`

5. Allegro Trends API (opcjonalnie):
- `ALLEGRO_TRENDS_ENABLED=true`
- `ALLEGRO_TRENDS_BASIC_AUTH_LOGIN=search-trends-ui`
- `ALLEGRO_TRENDS_BASIC_AUTH_PASSWORD=<basic auth password>`
- `ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_LOGIN=<technical account login>`
- `ALLEGRO_TRENDS_TECHNICAL_ACCOUNT_PASSWORD=<technical account password>`
- Opcjonalnie: `ALLEGRO_TRENDS_INTERVAL=day`, `ALLEGRO_TRENDS_MEASURES="VISIT,PV,OFFERS,GMV,DEALS"`, `ALLEGRO_TRENDS_TOP_ROWS=10`
- Raport pokazuje sekcje `Allegro Trends API (marketplace demand)` z metrykami dla top query movers.

6. Holidays/Ferie source:
- `HOLIDAYS_COUNTRY_CODE=PL`
- `HOLIDAYS_COUNTRY_CODE_MAP="PL:PL,CZ:CZ,SK:SK,HU:HU"`
- `HOLIDAYS_API_BASE_URL=https://openholidaysapi.org`
- `HOLIDAYS_LANGUAGE_CODE=PL`
- `HOLIDAYS_LANGUAGE_CODE_MAP="PL:PL,CZ:CS,SK:SK,HU:HU"`

7. Dodatkowe zrodla:
- `PAGESPEED_API_KEY` (zalecany: aktywuj `Chrome UX Report API` i `PageSpeed Insights API` w Google Cloud; bez klucza mozliwe limity 429)
- `GOOGLE_TRENDS_RSS_URL=https://trends.google.com/trending/rss?geo=PL`
- `GOOGLE_TRENDS_RSS_URL_MAP="PL:...,CZ:...,SK:...,HU:..."`
- `WEATHER_LATITUDE_MAP`, `WEATHER_LONGITUDE_MAP`, `WEATHER_LABEL_MAP` (koordynaty/etykiety per rynek)
- `MARKET_EVENTS_ENABLED=true`
- `MARKET_EVENTS_API_BASE_URL=https://api.gdeltproject.org/api/v2/doc/doc`
- `MARKET_EVENTS_TOP_ROWS=12`
- `NBP_API_BASE_URL=https://api.nbp.pl/api`
- `IMGW_WARNINGS_URL=https://danepubliczne.imgw.pl/api/data/warningsmeteo`

8. Google Drive / Google Docs:
- `GOOGLE_DRIVE_ENABLED=true`
- `GOOGLE_DRIVE_CLIENT_SECRET_PATH=client_secret_*.json` (OAuth client)
- `GOOGLE_DRIVE_TOKEN_PATH=.google_drive_token.json`
- `GOOGLE_DRIVE_FOLDER_NAME=\"SEO Weekly Reports\"`
- opcjonalnie `GOOGLE_DRIVE_FOLDER_ID` (jesli chcesz wymusic konkretny folder)

10. SEO team presentations (opcjonalnie, Google Drive):
- `SEO_PRESENTATIONS_ENABLED=true`
- `SEO_PRESENTATIONS_FOLDER_REFERENCE=<folder URL lub folder ID>`
- `SEO_PRESENTATIONS_MAX_FILES_PER_YEAR=20`
- `SEO_PRESENTATIONS_MAX_TEXT_FILES_PER_YEAR=8`

11. Historical continuity + status log (opcjonalnie, Google Drive):
- `HISTORICAL_REPORTS_ENABLED=true`
- `HISTORICAL_REPORTS_COUNT=3`
- `HISTORICAL_REPORTS_YOY_TOLERANCE_DAYS=28`
- `STATUS_LOG_ENABLED=true`
- `STATUS_FILE_REFERENCE=<URL/ID statusowego Google Sheeta>` (opcjonalnie; gdy puste, agent probuje znalezc status w folderze prezentacji albo globalnie po nazwie zawierajacej "status")
- `STATUS_MAX_ROWS=12`

12. Product trend sheets (Google Sheets, opcjonalnie ale zalecane):
- `PRODUCT_TRENDS_ENABLED=true`
- `PRODUCT_TRENDS_COMPARISON_SHEET_REFERENCE=<URL/ID arkusza do trendow YoY>` (gdy puste, agent uzyje `PRODUCT_TRENDS_CURRENT_SHEET_REFERENCE`)
- `PRODUCT_TRENDS_UPCOMING_SHEET_REFERENCE=<URL/ID arkusza upcoming trends>`
- `PRODUCT_TRENDS_CURRENT_SHEET_REFERENCE=<URL/ID arkusza current trends>`
- `PRODUCT_TRENDS_TOP_ROWS=12`
- `PRODUCT_TRENDS_HORIZON_DAYS=31`

Pierwsze uruchomienie Drive wymaga jednorazowej autoryzacji OAuth w przegladarce.
`secret.json` typu service account tez jest wspierany, ale moze zwracac `storageQuotaExceeded` bez dostepu do Shared Drive.

## Generacja tokenu Senuto

```bash
source .venv/bin/activate
weekly-seo-senuto-token --write-env
```

Komenda pobiera bearer token z `SENUTO_EMAIL` + `SENUTO_PASSWORD` i zapisuje do `.env` jako `SENUTO_TOKEN`.

## Automatyczna generacja GSC refresh token

Masz juz `client_secret_*.json`. Teraz wygeneruj refresh token:

```bash
source .venv/bin/activate
weekly-seo-gsc-token --write-env
```

Co robi komenda:

1. Otwiera consent screen Google.
2. Po autoryzacji pobiera `refresh_token`.
3. Zapisuje do `.env` jako `GSC_OAUTH_REFRESH_TOKEN`.

Uwaga: gdy Google nie zwroci refresh tokenu, cofnij dostep aplikacji i odpal ponownie z consent.

## Uruchomienie raportu bez crona

Najprosciej:

```bash
source .venv/bin/activate
set -a; source .env; set +a
weekly-seo-agent
```

Albo wrapper:

```bash
./scripts/run_report.sh
```

Run z konkretna data:

```bash
./scripts/run_report.sh --run-date 2026-02-10
```

## GitHub Actions (harmonogram)

Skonfigurowane workflow:
- `.github/workflows/weekly-report-agent.yml`  
  Uruchamia `weekly-seo-agent` w kazdy wtorek o `09:00` czasu PL (Europe/Warsaw, CET/CEST-safe).
- `.github/workflows/weekly-seo-news.yml`  
  Uruchamia `weekly-seo-news-agent` w kazdy poniedzialek o `09:00` czasu PL (Europe/Warsaw, CET/CEST-safe).

Wymagany sekret repo:
- `WEEKLY_SEO_ENV` - pelna zawartosc pliku `.env` (multiline secret).

Workflow zapisuje ten sekret do lokalnego `.env` w runtime i uruchamia agenta.

Opcjonalne sekrety na pliki credentials (JSON content, nie sciezka):
- `GSC_AUTH_CLIENT` (recommended; alias for OAuth client JSON used by GSC/Drive)
- `SERVICE_ACCOUNT_JSON` (recommended; service account JSON)
- `GSC_CREDENTIALS_JSON` (legacy)
- `GSC_OAUTH_CLIENT_SECRET_JSON` (legacy)
- `GOOGLE_DRIVE_CLIENT_SECRET_JSON` (legacy)
- `GOOGLE_DRIVE_TOKEN_JSON` (OAuth token cache; potrzebne, jesli Drive działa na OAuth bez interaktywnego logowania)

Jesli ustawione, workflow odtworzy je do `.secrets/*.json` i nadpisze odpowiednie zmienne path przez `GITHUB_ENV`.

Wynik:

- Generowany jest tylko plik `.docx`.
- Stare raporty nie sa kasowane; nadpisywane sa tylko pliki z tym samym prefiksem daty (`YYYY_MM_DD_`) przy ponownym runie tego samego dnia.
- Nazwa pliku zaczyna sie od daty: `YYYY_MM_DD_...`, np. `2026_02_10_pl_seo_weekly_report.docx`, `2026_02_10_cz_seo_weekly_report.docx`.
- Po wygenerowaniu `.docx` tworzony jest rowniez Google Doc o tej samej nazwie w folderze `SEO Weekly Reports` na Drive.

## Weekly SEO + GEO news digest (email)

Agent zbiera kluczowe newsy SEO i GEO z ostatniego pelnego tygodnia (pon-nd),
generuje streszczenie i wysyla je mailem przez Gmail API (service account z domain-wide delegation).

```bash
source .venv/bin/activate
set -a; source .env; set +a
weekly-seo-news-agent
```

Dry-run (bez wysylki maila):

```bash
weekly-seo-news-agent --dry-run
```

### OAuth bez domain-wide delegation (uzytkownik autoryzuje skrzynke)

Ten tryb nie wymaga dostepu admina. Uzywa zwyklego OAuth dla uzytkownika.

1. W Google Cloud Console wlacz Gmail API.
2. Utworz OAuth Client ID typu Desktop App.
3. Pobierz JSON i ustaw `GMAIL_OAUTH_CLIENT_SECRET_PATH`.
4. Wygeneruj refresh token:

```bash
weekly-seo-gmail-token --write-env
```

5. Ustaw w `.env`:
`GMAIL_AUTH_MODE=oauth`, `GMAIL_SENDER=<twoj email>`, `GMAIL_RECIPIENT=<email docelowy>`.

## Manager support agent (1:1 + review + readiness)

Agent analizuje notatki Google Docs dla konkretnej osoby i generuje raport managerski:

- tematy na 1:1,
- tematy do performance review,
- sygnaly gotowosci do awansu,
- mocne strony i ryzyka z evidence,
- merge tematow z pliku statusowego Google Sheets.

Minimalna konfiguracja `.env`:

- `PEOPLE_MANAGER_GOOGLE_CREDENTIALS_PATH` (lub fallback `GOOGLE_DRIVE_CLIENT_SECRET_PATH`)
- `GOOGLE_DRIVE_TOKEN_PATH=.google_drive_token.json`
- `PEOPLE_NOTES_FOLDER_REFERENCE=<URL/ID folderu z notatkami>`
- `PEOPLE_MANAGER_STATUS_SHEET_REFERENCE=<URL/ID statusowego Google Sheeta>` (opcjonalnie; fallback do `STATUS_FILE_REFERENCE`)
- `PEOPLE_MANAGER_STATUS_LLM_NAME_MAPPING=true` (LLM mapuje wpisy imie<->imie+nazwisko)

Uruchomienie:

```bash
source .venv/bin/activate
set -a; source .env; set +a
weekly-people-manager-agent --person "Imie"
```

Opcje:

- `--doc-reference <URL/ID>` (mozna podac wiele razy)
- `--max-docs 12`
- `--output outputs/people_manager/custom_report.md`
- `--no-llm` (tylko heurystyka, bez dopracowania LLM)
- `--status-sheet-reference <URL/ID>`
- `--status-max-topics 10`
- `--status-llm-name-mapping` / `--no-status-llm-name-mapping`

## Team manager agent (batch + upload do Drive)

Agent generuje plan rozmowy dla calego zespolu na bazie plikow osobowych (Google Docs)
i statusowego Google Sheeta, a nastepnie zapisuje wynik jako Google Docs w:
`SEO Team Data/<YYYY-MM-DD>/`.

Przykladowe uruchomienie:

```bash
source .venv/bin/activate
set -a; source .env; set +a
weekly-people-team-agent --exclude "Roksana"
```

Wynik:
- lokalne kopie markdown w `outputs/people_manager/team_plans/<YYYY_MM_DD>/`,
- Google Docs na Drive: po jednym pliku na osobe (nazwa pliku = imie osoby).

## Harmonogram (opcjonalnie)

```cron
CRON_TZ=Europe/Warsaw
0 10 * * 2 cd /path/to/repo && /path/to/repo/.venv/bin/bash -lc 'set -a; source .env; set +a; weekly-seo-agent'
```

## Testy

```bash
pytest
```
