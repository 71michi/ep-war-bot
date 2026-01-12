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


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == '':
        return default
    v = v.strip().lower()
    if v in ('1','true','yes','y','on'): return True
    if v in ('0','false','no','n','off'): return False
    return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default
