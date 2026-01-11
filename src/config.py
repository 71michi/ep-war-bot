import os
from dotenv import load_dotenv

load_dotenv()

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return int(v)

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v
