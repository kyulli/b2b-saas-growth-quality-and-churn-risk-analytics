from __future__ import annotations

import argparse
from pathlib import Path

from saas_growth_quality.public_comps import build_public_comps_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the processed public SaaS company-quarter panel from SEC snapshots.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    build_public_comps_panel(
        root / "data/public_comps/raw",
        root / "data/public_comps/company_universe.csv",
        root / "data/public_comps/processed",
    )


if __name__ == "__main__":
    main()
