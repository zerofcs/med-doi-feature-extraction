from __future__ import annotations

from typing import Callable, Dict

REGISTRY: Dict[str, Callable] = {}


def add(name: str, fn: Callable) -> None:
    REGISTRY[name] = fn


def get(name: str) -> Callable:
    return REGISTRY[name]


def names() -> list[str]:
    return list(REGISTRY.keys())

