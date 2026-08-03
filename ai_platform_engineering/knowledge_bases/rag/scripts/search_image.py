"""CLI wrapper for live image search against CAIPE's image collection.

The implementation lives in server.image_search. This wrapper adds the local
common/src and server/src folders to sys.path so the command works from a source
checkout without installing the package first.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for src_dir in (ROOT / "common" / "src", ROOT / "server" / "src"):
  sys.path.insert(0, str(src_dir))

from server.image_search import main


if __name__ == "__main__":
  raise SystemExit(main())
