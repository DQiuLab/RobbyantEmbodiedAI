"""Drop the page cache of large files (posix_fadvise DONTNEED; unprivileged, clean pages only).

Container memory accounting (cgroup memory.current) includes page cache. On AutoDL the trainer was
SIGKILLed twice (cgroup oom_kill stayed 0), both times around heavy file I/O (DCP save, snapshot merge);
we suspect (unconfirmed) the platform kills containers whose memory.current exceeds the 120 GiB quota. This keeps cache from DCPs/snapshots
we have just written or read out of the container's footprint. Dirty pages are written back first
(fdatasync) so they become droppable.

Usage: python src/evict_cache.py <dir> [<dir> ...] [--min-mb 64]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def large_files(roots: list[Path], min_bytes: int):
    for root in roots:
        if root.is_file():
            if root.stat().st_size >= min_bytes:
                yield root
            continue
        for dirpath, _, names in os.walk(root):
            for n in names:
                p = Path(dirpath) / n
                try:
                    if p.is_file() and p.stat().st_size >= min_bytes:
                        yield p
                except OSError:
                    continue


def evict(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        try:
            os.fdatasync(fd)
        except OSError:
            pass
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--min-mb", type=int, default=64)
    a = ap.parse_args(argv)
    n = sum(evict(p) for p in large_files(a.roots, a.min_mb * 2**20))
    print(f"evicted page cache of {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
