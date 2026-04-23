# EP War Bot v3.4.64 — points / mode / average indicators

Odtworzona pełna paczka projektu na podstawie ustaleń z historii projektu.
Ta wersja skupia się na zmianach widoku szczegółów wojny:

- bardziej kompaktowy widok szczegółów wojny
- ikonka trybu wojennego obok wyniku gracza
- oznaczenie powyżej / poniżej średniej
- cienka, przerywana linia średniej z podpisem
- prosty backend FastAPI + frontend React/Vite

## Uruchomienie

### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend domyślnie pobiera dane z `http://localhost:8000/api/wars`.

## Uwaga
To jest rekonstrukcja pakietu z naciskiem na opisane zmiany UI. Nie zawiera pełnej
logiki Discord/OCR z wcześniejszych iteracji projektu.
