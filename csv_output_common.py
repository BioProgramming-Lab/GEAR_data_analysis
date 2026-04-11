from __future__ import annotations

from pathlib import Path

import pandas as pd


def clean_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, [col for col in df.columns if not str(col).startswith("Unnamed:")]].copy()


def load_export_dataframe(path: Path) -> pd.DataFrame:
    return clean_export_dataframe(pd.read_csv(path))


def combine_export_tables(
    output_path: Path,
    tables: list[dict[str, object]],
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepared_frames: list[pd.DataFrame] = []
    for table in tables:
        if "dataframe" in table:
            df = clean_export_dataframe(table["dataframe"])
        elif "path" in table:
            df = load_export_dataframe(Path(table["path"]))
        else:
            raise ValueError("Each table must provide either 'dataframe' or 'path'.")

        metadata = dict(table.get("metadata", {}))
        for key, value in reversed(list(metadata.items())):
            df.insert(0, key, value)
        prepared_frames.append(df)

    combined_df = pd.concat(prepared_frames, ignore_index=True, sort=False)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")
    return combined_df


def remove_files(paths: list[Path]) -> None:
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()
            print(f"Removed intermediate CSV: {path}")
