#!/usr/bin/env python3
"""
price_loader.py
Dynamic raw-material price / CO2-factor loader.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_DATE_TOKEN_RE = re.compile(
    r"(\d{1,2}[_\-][A-Za-z]{3,9}[_\-]\d{4}|\d{4}[_\-]\d{1,2}[_\-]\d{1,2}|\d{1,2}[_\-]\d{1,2}[_\-]\d{4})"
)

_DATE_FORMATS = [
    "%d_%B_%Y", "%d_%b_%Y", "%d-%B-%Y", "%d-%b-%Y",
    "%Y_%m_%d", "%Y-%m-%d",
    "%d_%m_%Y", "%d-%m-%Y",
]


def _parse_date_from_filename(filename: str) -> Optional[datetime]:
    match = _DATE_TOKEN_RE.search(filename)
    if not match:
        return None
    token = match.group(1)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    return None


def _resolve_file_date(path: Path) -> datetime:
    dt = _parse_date_from_filename(path.name)
    if dt is not None:
        return dt
    warnings.warn(
        f"Could not parse a date from filename '{path.name}'. "
        "Falling back to file-modified-time. Rename the file to include "
        "a date, e.g. '17_April_2026_Cost.csv', for a reliable audit trail."
    )
    return datetime.fromtimestamp(path.stat().st_mtime)


def find_latest_file(folder: Path, keyword: str) -> Optional[Path]:
    folder = Path(folder)
    if not folder.exists():
        return None

    candidates = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".csv" and keyword.lower() in f.name.lower()
    ]
    if not candidates:
        return None

    dated = [(_resolve_file_date(f), f) for f in candidates]
    dated.sort(key=lambda pair: pair[0])
    return dated[-1][1]


def load_price_table(
    folder,
    keyword: str,
    value_col: str,
    materials: list[str],
    fallback: Optional[dict] = None,
    material_col: str = "Material",
    date_col: str = "Effective_Date",
) -> tuple[dict, dict]:
    fallback = fallback or {}
    latest = find_latest_file(folder, keyword)

    if latest is None:
        warnings.warn(
            f"No dated '{keyword}' CSV found in '{folder}'. "
            "Using fallback values (e.g. from metadata.json)."
        )
        values = {m: float(fallback.get(m, 0.0)) for m in materials}
        return values, {"file": None, "as_of": None, "source": "fallback"}

    df = pd.read_csv(latest)
    if material_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"'{latest.name}' must contain columns '{material_col}' and "
            f"'{value_col}'. Found: {list(df.columns)}"
        )
    table = dict(zip(df[material_col].astype(str), df[value_col].astype(float)))

    filename_date = _resolve_file_date(latest)
    if date_col in df.columns and not df[date_col].isna().all():
        try:
            internal_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not internal_dates.empty:
                internal_date = internal_dates.max()
                if abs((internal_date - filename_date).days) > 0:
                    warnings.warn(
                        f"'{latest.name}': filename date ({filename_date.date()}) "
                        f"differs from internal '{date_col}' ({internal_date.date()}). "
                        "Using the filename date to determine recency."
                    )
        except Exception:
            pass

    values = {}
    missing = []
    for m in materials:
        if m in table:
            values[m] = float(table[m])
        else:
            missing.append(m)
            values[m] = float(fallback.get(m, 0.0))
    if missing:
        warnings.warn(
            f"'{latest.name}' is missing values for {missing}; "
            "used fallback values for those materials."
        )

    return values, {"file": latest, "as_of": filename_date, "source": "file"}
