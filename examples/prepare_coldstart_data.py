"""Create wide and grouped cold-start panels from repository ETTh1 data.

The cold item A has a short history while donor items B and C have longer
histories. All items share one forecast origin and horizon. This script is the
only place that truncates history; production model code receives the history
that is actually present in its input data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "tests" / "data" / "ETTh1.csv"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "coldstart" / "data"
GROUPED_FILENAME = "etth1_coldstart_grouped.csv"
WIDE_FILENAME = "etth1_coldstart_wide.csv"
SERIES = {"A": "HUFL", "B": "MUFL", "C": "HULL"}
DEFAULT_SHORT_HISTORY = 96
DEFAULT_DONOR_HISTORY = 672
DEFAULT_HORIZON = 24


def prepare(
    input_path: Path,
    output_dir: Path,
    *,
    short_history: int,
    donor_history: int,
    horizon: int,
    origin: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Write equivalent grouped and wide views of one cold-start panel."""
    source = pd.read_csv(input_path, parse_dates=["date"]).sort_values("date")
    origin_pos = _origin_position(source, origin, horizon)
    histories = {"A": short_history, "B": donor_history, "C": donor_history}
    _validate_lengths(source, histories, origin_pos, horizon)

    grouped = _build_grouped(source, histories, origin_pos, horizon)
    wide = _build_wide(source, histories, origin_pos, horizon)

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_dir / GROUPED_FILENAME, index=False)
    wide.to_csv(output_dir / WIDE_FILENAME, index=False)
    return grouped, wide


def _validate_lengths(
    source: pd.DataFrame,
    histories: dict[str, int],
    origin_pos: int,
    horizon: int,
) -> None:
    if min(histories.values()) <= 0 or horizon <= 0:
        raise ValueError("History lengths and horizon must be positive.")
    if origin_pos < max(histories.values()):
        raise ValueError(
            f"Origin leaves only {origin_pos} history rows; "
            f"donor_history requires {max(histories.values())}."
        )
    if origin_pos + horizon > len(source):
        raise ValueError("Origin does not leave enough future truth for the horizon.")


def _build_grouped(
    source: pd.DataFrame,
    histories: dict[str, int],
    origin_pos: int,
    horizon: int,
) -> pd.DataFrame:
    parts = []
    for item_id, source_col in SERIES.items():
        history = histories[item_id]
        item = source.iloc[origin_pos - history : origin_pos + horizon][
            ["date", source_col, "OT"]
        ].copy()
        item = item.rename(columns={source_col: "target"})
        item.insert(1, "item_id", item_id)
        parts.append(item)
    return pd.concat(parts, ignore_index=True)


def _build_wide(
    source: pd.DataFrame,
    histories: dict[str, int],
    origin_pos: int,
    horizon: int,
) -> pd.DataFrame:
    """Build a rectangular frame; unavailable A history is represented by NaN."""
    start = origin_pos - max(histories.values())
    wide = source.iloc[start : origin_pos + horizon][["date", "OT"]].copy()
    for item_id, source_col in SERIES.items():
        wide[item_id] = source.iloc[start : origin_pos + horizon][source_col].to_numpy()
        unavailable = max(histories.values()) - histories[item_id]
        if unavailable:
            wide.iloc[:unavailable, wide.columns.get_loc(item_id)] = float("nan")
    return wide[["date", "A", "B", "C", "OT"]]


def _origin_position(source: pd.DataFrame, origin: str | None, horizon: int) -> int:
    if origin is None:
        return len(source) - horizon
    timestamp = pd.Timestamp(origin)
    positions = source.index[source["date"] == timestamp]
    if len(positions) != 1:
        raise ValueError(f"Origin {origin!r} must match exactly one ETTh1 timestamp.")
    return int(source.index.get_loc(positions[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--short-history", type=int, default=DEFAULT_SHORT_HISTORY)
    parser.add_argument("--donor-history", type=int, default=DEFAULT_DONOR_HISTORY)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--origin", default=None)
    args = parser.parse_args()
    grouped, wide = prepare(
        args.input,
        args.output_dir,
        short_history=args.short_history,
        donor_history=args.donor_history,
        horizon=args.horizon,
        origin=args.origin,
    )

    grouped_counts = grouped.groupby("item_id").size().to_dict()
    truth = grouped.groupby("item_id").tail(args.horizon)
    print(f"Wrote {args.output_dir / GROUPED_FILENAME}: {grouped_counts}")
    print(
        f"Wrote {args.output_dir / WIDE_FILENAME}: {len(wide)} rows; "
        f"observed targets={wide[list(SERIES)].count().to_dict()}"
    )
    print(f"Common truth window: {truth['date'].min()} -> {truth['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
