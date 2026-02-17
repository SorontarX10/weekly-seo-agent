# Weekly Reporting Agent

Ten folder jest **punktem wejścia funkcjonalnym** do agenta raportowania tygodniowego SEO (PL/CZ/SK/HU).  
Możesz go wysłać komuś jako "tu jest agent + co robi + jak uruchomić".

## 1) Co robi ten agent

Agent:
- pobiera dane z GSC, Senuto, pogody, newsów, eventów i innych źródeł,
- buduje raport tygodniowy (Executive Summary + analiza przyczynowa + driver scoreboard),
- zapisuje raport lokalnie jako `.docx`,
- opcjonalnie publikuje raport na Google Drive jako **Google Doc**.

## 2) Gdzie jest kod agenta

Ten folder jest "wejściem" i dokumentacją. Implementacja znajduje się tutaj:
- `weekly_seo_agent/weekly_reporting_agent/main.py` - orchestrator batch run (multi-country, równolegle),
- `weekly_seo_agent/weekly_reporting_agent/workflow.py` - pipeline zbierania danych i analizy,
- `weekly_seo_agent/weekly_reporting_agent/reporting.py` - generowanie sekcji raportu + render `.docx`,
- `weekly_seo_agent/weekly_reporting_agent/clients/google_drive_client.py` - publikacja `.docx` do Google Docs,
- `weekly_seo_agent/weekly_reporting_agent/config.py` - konfiguracja z `.env`.

## 3) Jak uruchomić

Najprościej:

```bash
./scripts/run_report.sh
```

lub przez wrapper z tego folderu:

```bash
./agents/weekly-reporting-agent/run_weekly_reporting.sh
```

Z datą ręczną:

```bash
./scripts/run_report.sh --run-date 2026-02-17
```

## 4) Co musi być w `.env`

Minimum dla działania raportu:
- GSC credentials (`GSC_CREDENTIALS_PATH` albo OAuth refresh token flow),
- model LLM (`OPENAI_API_KEY` / GAIA config zależnie od środowiska),
- `REPORT_COUNTRIES=PL,CZ,SK,HU` (lub inny zestaw),
- `GSC_SITE_URL_MAP=PL:https://allegro.pl,CZ:https://allegro.cz,SK:https://allegro.sk,HU:https://allegro.hu`.

Dla publikacji do Google Docs:
- `GOOGLE_DRIVE_ENABLED=true`
- `GOOGLE_DRIVE_CLIENT_SECRET_PATH=...` (OAuth client JSON albo service account JSON)
- `GOOGLE_DRIVE_TOKEN_PATH=.google_drive_token.json`
- `GOOGLE_DRIVE_FOLDER_NAME=SEO Weekly Reports`
- opcjonalnie: `GOOGLE_DRIVE_FOLDER_ID=...` (jeśli chcesz pisać do konkretnego folderu)

## 5) Jak dokładnie działa zapis do Google Docs

Przepływ jest deterministyczny i wygląda tak:

1. **Pipeline kończy analizę** i zwraca tekst raportu (`final_report` albo `markdown_report`) w `weekly_seo_agent/weekly_reporting_agent/main.py`.
2. Agent buduje nazwę pliku:
   - `YYYY_MM_DD_<country>_seo_weekly_report.docx`
3. Tekst raportu jest renderowany do DOCX przez:
   - `weekly_seo_agent/weekly_reporting_agent/reporting.py` -> `write_docx(path, title, content)`
4. Jeśli `config.google_drive_upload_enabled` jest `true`, agent inicjalizuje:
   - `GoogleDriveClient(...)` z `weekly_seo_agent/weekly_reporting_agent/clients/google_drive_client.py`
5. Dla każdego wygenerowanego pliku `.docx` wywoływane jest:
   - `upload_docx_as_google_doc(local_docx_path)`
6. Wewnątrz `upload_docx_as_google_doc`:
   - ładowane są credentials (`OAuth` albo `service account`),
   - wyszukiwany / tworzony jest folder docelowy (`_find_or_create_folder`),
   - usuwane są istniejące Google Docs o tej samej nazwie (`_delete_existing_docs`) - dzięki temu ten sam dzień/kraj nie duplikuje dokumentu,
   - wykonywany jest upload `MediaFileUpload` z MIME DOCX,
   - Drive API tworzy plik z `mimeType=application/vnd.google-apps.document` (konwersja do Google Docs),
   - zwracany jest `webViewLink` i logowany na stdout.

### Ważne zachowanie
- Lokalny `.docx` **zostaje** w `output_dir`.
- Na Drive agent utrzymuje **1 dokument na nazwę** (czyli nadpisuje logicznie przez delete+create dla tej samej nazwy).

## 6) Jakie funkcje są "w tym agencie"

Najważniejsze capability:
- tygodniowe okna dat (WoW + YoY aligned 52W),
- analiza brand vs non-brand,
- analiza klastrów zapytań i feature splitów GSC,
- korelacja wyników z kampaniami (trade plan), pogodą, eventami i SEO updates,
- raport menedżerski + analityczny w jednym dokumencie,
- publikacja do Google Docs.

## 7) Troubleshooting (najczęstsze)

- Brak Google Doc mimo wygenerowanego `.docx`:
  - sprawdź `GOOGLE_DRIVE_ENABLED=true` i poprawność credentials,
  - sprawdź, czy konto ma dostęp do folderu docelowego (lub użyj `GOOGLE_DRIVE_FOLDER_ID`).
- Błąd quota w Drive:
  - dla service account użyj folderu Shared Drive z write access,
  - albo użyj OAuth user credentials.
- GSC nie zwraca danych per rynek:
  - sprawdź `GSC_SITE_URL_MAP` i uprawnienia konta technicznego do każdej właściwości.

## 8) Zakres folderu

Ten folder celowo nie duplikuje kodu źródłowego.  
Jego rola: **jedno miejsce do udostępnienia i onboardingu** osoby, która ma uruchomić/utrzymać agenta weekly reporting.
