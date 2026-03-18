"""Inference entrypoint placeholder for phase-1 experiments."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    raise NotImplementedError("Inference pipeline is not implemented yet.")


if __name__ == "__main__":
    main()
