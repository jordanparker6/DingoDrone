from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class Command:
    name: str
    keywords: List[str]

COMMANDS = {
    "takeoff": Command(name="takeoff", keywords=["take off"]),
    "land": Command(name="land", keywords=["land"]),
    "spin": Command(name="spin", keywords=["spin"]),
}
