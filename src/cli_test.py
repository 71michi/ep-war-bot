import sys
from pathlib import Path

from .openai_parser import parse_war_from_images
from .config import env_str
from .bot import format_message

def main():
    if len(sys.argv) < 3:
        print("Użycie: python -m src.cli_test chat.png wojna.png")
        raise SystemExit(2)

    paths = [Path(sys.argv[1]), Path(sys.argv[2])]
    images = [p.read_bytes() for p in paths]

    model = env_str("OPENAI_MODEL", "gpt-4o-mini")
    summary, players, debug, meta = parse_war_from_images(images, model=model)

    if not summary or not players:
        print("Nie udało się wyciągnąć danych.")
        print(debug)
        return

    print(format_message(summary, players, meta=meta))

if __name__ == "__main__":
    main()
