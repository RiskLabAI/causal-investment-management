"""Download public sources and recreate local inputs for the Julia notebooks."""

from __future__ import annotations

import argparse
import importlib
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_DIRECTORY = Path(__file__).resolve().parent / "input_generators"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GENERATOR_DIRECTORY))

from book_support import DATASETS, ensure_dataset
from input_common import save_exact


GENERATORS = {
    "1": "prepare_cim_chapter01_inputs",
    "2": "prepare_cim_chapter02_inputs",
    "3": "prepare_cim_chapter03_inputs",
}


def prepare_public_data() -> None:
    paths = {name: ensure_dataset(ROOT, name) for name in DATASETS}
    native_directory = ROOT / "notebooks" / "julia" / "data"
    for name, destination_name in (
        ("binance_btcusdt", "binance_btcusdt.csv"),
        ("french_49_industries", "french_49_industries.csv"),
    ):
        with zipfile.ZipFile(paths[name]) as archive:
            content = archive.read(archive.namelist()[0])
        save_exact(native_directory / destination_name, content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", action="append", choices=[*GENERATORS, "all"], default=[])
    parser.add_argument("--public-data-only", action="store_true")
    arguments = parser.parse_args()
    prepare_public_data()
    if arguments.public_data_only:
        return
    selected = list(GENERATORS) if not arguments.chapter or "all" in arguments.chapter else arguments.chapter
    for chapter in selected:
        generator = importlib.import_module(GENERATORS[chapter])
        if chapter == "2":
            for section in ("backdoor", "frontdoor", "iv"):
                generator.main(section)
        else:
            generator.main()


if __name__ == "__main__":
    main()
