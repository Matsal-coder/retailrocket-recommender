"""Loaders for the RetailRocket dataset."""

from pathlib import Path

import pandas as pd


def load_events(path: str | Path) -> pd.DataFrame:
    """Load the RetailRocket events CSV file.

    Parameters
    ----------
    path:
        Path to the RetailRocket `events.csv` file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw events.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the loaded CSV file is empty.
    """
    events_path = Path(path)

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    try:
        events = pd.read_csv(events_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Events file is empty: {events_path}") from exc

    if events.empty:
        raise ValueError(f"Events file is empty: {events_path}")

    return events
