"""Backs up and restores the state that matters: the MLflow store and the prediction log.

    python backup.py                          # -> backups/observatory-20261120-031500.tar.gz
    python backup.py --keep 7                 # and delete all but the 7 newest backups
    python backup.py restore backups/observatory-20261120-031500.tar.gz

What's in a backup:
  models/mlflow.db       runs, metrics, model versions and the champion/challenger aliases
  models/mlartifacts/    model files and their drift reference data
  logs/predictions.db    every scored client, and the shadow comparisons

The SQLite files are copied with SQLite's online backup API, so a backup taken
while the API is serving is consistent. Restore with the API stopped, then start
it again: it serves whichever version the restored champion alias points at.

In docker compose (the store lives in named volumes):
    docker compose run --rm -v "$PWD/backups:/app/backups" api python backup.py
"""
import argparse
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

DATABASES = ("models/mlflow.db", "logs/predictions.db")
DIRECTORIES = ("models/mlartifacts",)


def _copy_sqlite(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def create(root=".", out_dir="backups", keep=None, now=None):
    """Writes one .tar.gz backup and returns its path."""
    root, out_dir = Path(root), Path(root) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"observatory-{stamp}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(path, "w:gz") as tar:
        for name in DATABASES:
            if (root / name).exists():
                copy = Path(tmp) / name
                _copy_sqlite(root / name, copy)
                tar.add(copy, arcname=name)
        for name in DIRECTORIES:
            if (root / name).exists():
                tar.add(root / name, arcname=name)
    if keep:
        for old in sorted(out_dir.glob("observatory-*.tar.gz"))[:-keep]:
            old.unlink()
    return path


def _expected(name):
    return name in DATABASES or any(name == d or name.startswith(d + "/") for d in DIRECTORIES)


def restore(archive, root="."):
    """Replaces the store and the log with the archive's. Run with the API stopped."""
    root = Path(root)
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        if not names or not all(_expected(n) for n in names):
            raise ValueError(f"{archive} is not an observatory backup")
        for name in DIRECTORIES:
            if any(n == name or n.startswith(name + "/") for n in names):
                shutil.rmtree(root / name, ignore_errors=True)
        for name in DATABASES:
            for suffix in ("", "-wal", "-shm"):
                (root / (name + suffix)).unlink(missing_ok=True)
        tar.extractall(root, filter="data")  # "data" refuses absolute paths, links out and devices
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", default="create", choices=["create", "restore"])
    ap.add_argument("archive", nargs="?", help="for restore: the .tar.gz to restore")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="backups", help="where backups go (relative to --root)")
    ap.add_argument("--keep", type=int, help="keep only this many newest backups")
    args = ap.parse_args(argv)
    if args.action == "restore":
        if not args.archive:
            ap.error("restore needs the archive to restore")
        names = restore(args.archive, args.root)
        print(f"restored {len(names)} files from {args.archive}; start the API again to serve them")
    else:
        print(create(args.root, args.out, args.keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
