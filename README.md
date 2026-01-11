# EP War Bot (Discord) – Empires & Puzzles

Bot na Discorda do automatycznego podsumowania wojny z gry **Empires & Puzzles** na podstawie **2 screenów**:

1. ranking z chatu: **„Najlepsi atakujący w wojnach”**
2. ekran podsumowania wojny (wynik + tryb)

## Najważniejsze cechy

- **2-fazowe parsowanie** listy z chatu:
  - 1 request (szybko i tanio)
  - jeśli walidacja wykryje problem → automatycznie uruchamia się tryb „robust” (slicing + overlap)
  - jeśli dalej jest problem → fallback do większej liczby slice’ów
- **Walidacja po rankach**:
  - wykrywa duplikaty oraz brakujące pozycje
  - dodatkowo próbuje odczytać **największy widoczny numer rankingu** z dołu screena i waliduje względem niego
- **Ostrzeżenie w wiadomości** jeśli są braki/duplikaty.
- **Manualne poprawki przez reply** (bez ponownego wrzucania screenów):
  - odpowiadasz na wiadomość bota np. `23 ropuch13 250`
  - bot edytuje swoją wiadomość i uzupełnia / poprawia wpis
- **Roster-only output**: finalne nicki są mapowane do `roster.json` (aliasy + fuzzy), a jeśli nie ma pewności → `UNKNOWN`.

## Konfiguracja

1) Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

2) Ustaw zmienne środowiskowe (najprościej skopiuj `.env.example` → `.env`):

- `DISCORD_TOKEN` – token bota
- `WATCH_CHANNEL_ID` – ID kanału, na którym bot ma reagować
- `OPENAI_API_KEY` – klucz OpenAI
- `OPENAI_MODEL` – domyślnie `gpt-4o-mini`
- `OPENAI_TIMEOUT_SECONDS` – domyślnie 90
- `OPENAI_CHAT_SLICES` – liczba slice’ów w trybie robust (domyślnie 4)
- `ROSTER_PATH` – domyślnie `roster.json`
- `ALIASES_PATH` – domyślnie `aliases.json`

## Jak używać na Discordzie

1) Wklej na obserwowany kanał **2 obrazki** (attachmenty) w jednej wiadomości:
- 1) ranking z chatu „Najlepsi atakujący w wojnach”
- 2) podsumowanie wojny

2) Bot odpowie podsumowaniem i listą:

```
[01] Nick — 307
[02] Nick — 297
...
```

3) Jeśli bot wykryje problem, dopisze warning, np.:

> ⚠️ Brak odczytu pozycji (1–30): 23.

### Manualne poprawki (reply)

Odpowiedz (reply) na wiadomość bota jedną lub kilkoma liniami:

```
23 ropuch13 250
30 max72 118
```

Bot znormalizuje nicki, zaktualizuje swoją wiadomość (edit) i doda reakcję ✅.

## Lokalny test parsera (CLI)

```bash
python -m src.cli_test chat.png war.png
```

## Pliki danych

- `roster.json` – lista kanonicznych nicków
- `aliases.json` – mapowania aliasów (exact/canonical)

