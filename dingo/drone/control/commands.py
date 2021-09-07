from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class Command:
    name: str
    keywords: List[str]
    key: str

COMMANDS = {
    "takeoff": Command(name="takeoff", keywords=["take off"], key="enter"),
    "land": Command(name="land", keywords=["land"], key="enter"),
    "spin": Command(name="spin", keywords=["spin"], key="z"),
    "backflip": Command(name="backflip", keywords=["back flip"], key="b"),
    "throwfly": Command(name="throwfly", keywords=["get ready"], key="tab")
}
