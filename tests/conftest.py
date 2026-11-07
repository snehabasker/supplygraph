from __future__ import annotations

import pytest

# Real, check-digit-valid LEIs (Apple, NVIDIA, Rosneft) so fixtures pass the same
# validation as production data.
APPLE = "HWUPKR0MPOU8FGXBT394"
NVIDIA = "549300S4KLFTLO7GSQ80"
ROSNEFT = "253400JT3MQWNDKMJE44"


def make_record(lei: str, name: str, country: str = "US") -> dict:
    return {
        "type": "lei-records",
        "id": lei,
        "attributes": {
            "lei": lei,
            "entity": {
                "legalName": {"name": name, "language": "en"},
                "legalAddress": {"city": "Somewhere", "country": country},
                "headquartersAddress": {"city": "Somewhere", "country": country},
                "jurisdiction": country,
                "category": "GENERAL",
                "legalForm": {"id": "XXXX", "other": None},
                "status": "ACTIVE",
            },
            "registration": {"status": "ISSUED", "lastUpdateDate": "2026-01-01T00:00:00Z"},
        },
    }


@pytest.fixture
def records() -> dict[str, dict]:
    return {
        APPLE: make_record(APPLE, "Apple Inc."),
        NVIDIA: make_record(NVIDIA, "NVIDIA CORPORATION"),
        ROSNEFT: make_record(ROSNEFT, "Rosneft Oil Company", country="RU"),
    }
