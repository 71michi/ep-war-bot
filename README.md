# ep-war-bot

## Manualne poprawki na Discordzie

Jeśli bot wyświetli sekcję „Wymagane poprawki”, możesz odpisać (Reply) na wiadomość bota:

- `23 ropuch13 250` → ustaw nick i punkty
- `23 Legendarny` → ustaw nick, zachowaj istniejące punkty

### Dodawanie nowych członków do rosteru (ADDROSTER)

Jeśli ktoś nie jest w `roster.json`, możesz dodać go bez deploya:

- `ADDROSTER Krati`
- `ADDROSTER Krati, NowyGracz`

Bot dopisze wpisy do pliku `roster_overrides.json` i (jeśli to możliwe) od razu przeliczy wynik pod wiadomością.