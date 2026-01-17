# ep-war-bot

## Manualne poprawki na Discordzie

## Workflow dodawania wojny do strony (LISTWAR → poprawki → ADDWAR)

1. Ktoś wrzuca na kanał 2 screeny z wojny.
2. Ktoś odpowiada (Reply) na tę wiadomość komendą `LISTWAR`.
3. Bot generuje listę wyników w trybie **DRAFT**.
4. Jeżeli coś jest nie tak, poprawiasz Reply na wiadomość bota: `23 Nick 250` lub `23 Nick` (punkty zostają).
5. Gdy lista jest poprawna, odpowiadasz na wiadomość bota komendą `ADDWAR` — dopiero wtedy wojna trafia na stronę (**CONFIRMED**).
6. Usunięcie wojny ze strony: `REMOVEWAR <ID>`.

Jeśli bot wyświetli sekcję „Wymagane poprawki”, możesz odpisać (Reply) na wiadomość bota:

- `23 ropuch13 250` → ustaw nick i punkty
- `23 Legendarny` → ustaw nick, zachowaj istniejące punkty

### Dodawanie nowych członków do rosteru (ADDROSTER)

Jeśli ktoś nie jest w `roster.json`, możesz dodać go bez deploya:

- `ADDROSTER Krati`
- `ADDROSTER Krati, NowyGracz`

Bot dopisze wpisy do pliku `roster_overrides.json` i (jeśli to możliwe) od razu przeliczy wynik pod wiadomością.


## Trwałość na darmowym hostingu (Render Free) — bez utraty progresu

Na darmowych planach (np. Render Free) system plików bywa **ulotny** (po restarcie / uśpieniu / redeployu pliki JSON mogą się wyzerować). Żeby **nie tracić historii wojen i zmian rosteru**, bot potrafi zapisywać snapshoty do prywatnego kanału Discord (jako przypięte wiadomości z załącznikiem).

### Jak włączyć

1. Utwórz prywatny kanał np. `#warbot-storage` (widoczny tylko dla adminów i bota).
2. Nadaj botowi uprawnienia w tym kanale: **View Channel**, **Read Message History**, **Send Messages**, **Attach Files**, **Manage Messages** (dla pin/unpin).
3. Ustaw na hoście zmienne środowiskowe:
   - `DISCORD_STORAGE_CHANNEL_ID=...` (ID kanału storage)
   - opcjonalnie `DISCORD_PERSIST_MAX_BYTES=7000000` (limit, po którym bot gzipuje plik przed wysłaniem)

Od tej pory bot będzie:
- przy starcie próbował odtworzyć: `wars_store.json`, `roster_overrides.json`, `roster_removed.json` z przypiętych wiadomości,
- po każdej operacji `ADDWAR/REMOVEWAR` oraz `ADDROSTER/REMOVEROSTER` aktualizował snapshot.

## Batch test (lokalne testowanie wielu screenów)

Jeżeli chcesz przetestować 10–50 screenów "hurtowo" tym samym modelem i tym samym pipeline'm co bot na Discordzie, użyj skryptu `src/batch_test.py`.

### 1) Przygotuj `.env`

Upewnij się, że masz ustawione co najmniej:
- `OPENAI_API_KEY=...`
- (opcjonalnie) `OPENAI_MODEL=gpt-4o-mini`

### 2) Struktura input

**Opcja A (polecana): podfolder = jeden test**

```
input/
  test01/
    chat.png
    summary.png
  test02/
    1.png
    2.png
```

**Opcja B: folder płaski (grupowanie po prefiksie przed `_` lub `-`)**

```
input/
  test01_chat.png
  test01_summary.png
  test02_a.png
  test02_b.png
```

### 3) Uruchomienie

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.batch_test --input ./input --out ./results
# opcjonalnie:
python -m src.batch_test --input ./input --out ./results --model gpt-4o-mini
```

### 4) Wyniki

W folderze `results/` powstaną:
- `index.csv` – szybkie podsumowanie wszystkich testów (OK / UNKNOWN / poza rosterem / braki)
- `<case>.post.txt` – dokładnie to, co bot by wysłał na Discordzie
- `<case>.raw.json` – surowe dane (summary/players/parsed_debug)
- `<case>.log.txt` – szczegółowy log krok-po-kroku dla danego testu
