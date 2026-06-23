"""Download the bundled tutorial cutouts.

Run via:

    python -m euclid_phot.examples_data

or, from inside Python:

    from euclid_phot import examples_data
    examples_data.fetch()

Self-contained (no package imports) so it runs before Tractor is installed.
Pulls a ~150 MB tarball from the project's GitHub Release page (50 and 200
arcsec EDF-N demo cutouts + PSFs + unWISE mosaics) and extracts it to
``examples/data/``.

Environment variables:
    EUCLID_PHOT_DATA_URL   override the default release URL
    EUCLID_PHOT_DATA_DIR   override the default extraction directory
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_DATA_DIR = Path("examples/data")
TUTORIAL_DATA_URL = (
    "https://github.com/nchartab/euclid-forced-photometry/"
    "releases/download/v0.4.1-data/tutorial-data-v0.4.1.tar.gz"
)
TUTORIAL_DATA_SHA256: str | None = (
    "a8c4235d34423e741902abd42fadf85b9de2286ee8422278d50528b4049cd92a"
)


def _sha256_of(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(data_dir: Path | str = DEFAULT_DATA_DIR,
          url: str | None = None,
          *,
          force: bool = False) -> Path:
    """Download and extract the bundled tutorial data.

    Returns the path to the populated ``examples/data/`` directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = (url
           or os.environ.get("EUCLID_PHOT_DATA_URL")
           or TUTORIAL_DATA_URL)
    tar_path = data_dir / "tutorial-data.tar.gz"

    if tar_path.exists() and not force:
        print(f"using existing {tar_path}")
    else:
        # Atomic download via .part rename.
        tmp = tar_path.with_suffix(tar_path.suffix + ".part")
        print(f"downloading {url} -> {tar_path}")
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(tar_path)
        print(f"  wrote {tar_path.stat().st_size / 1e6:.1f} MB")

    if TUTORIAL_DATA_SHA256:
        actual = _sha256_of(tar_path)
        if actual != TUTORIAL_DATA_SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch for {tar_path}:\n"
                f"  expected {TUTORIAL_DATA_SHA256}\n"
                f"  got      {actual}"
            )
        print("  checksum OK")
    else:
        print("  (no checksum configured; skipping verification)")

    with tarfile.open(tar_path, "r:gz") as tar:
        # tarfile filter="data" requires Python 3.12.
        if sys.version_info >= (3, 12):
            tar.extractall(data_dir, filter="data")
        else:
            tar.extractall(data_dir)
    print(f"extracted to {data_dir}")
    return data_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m euclid_phot.examples_data",
        description="Download the bundled euclid_phot tutorial data.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("EUCLID_PHOT_DATA_DIR", str(DEFAULT_DATA_DIR)),
        help="Destination directory (default: examples/data)")
    parser.add_argument("--url", default=None,
                        help="Override the default release URL")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the tarball already exists")
    args = parser.parse_args(argv)

    try:
        fetch(Path(args.data_dir), url=args.url, force=args.force)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "\nThe data bundle could not be downloaded. The notebooks will "
            "still work; anything missing is fetched live from IRSA/S3 and "
            "cached under examples/data/, so the first run is slower.",
            file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
