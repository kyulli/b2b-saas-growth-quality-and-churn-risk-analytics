from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


SEC_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def download_company_facts(universe_path: Path, output_dir: Path, *, user_agent: str, refresh: bool = False) -> None:
    """Download official SEC Company Facts JSON snapshots with a declared user agent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(universe_path, dtype={"cik": str})
    for company in universe.itertuples(index=False):
        cik = str(company.cik).zfill(10)
        target = output_dir / f"CIK{cik}.json"
        if target.exists() and not refresh:
            continue
        request = Request(
            SEC_BASE_URL.format(cik=cik),
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
        )
        with urlopen(request, timeout=30) as response:
            target.write_bytes(response.read())
        time.sleep(0.15)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEC Company Facts snapshots for the public SaaS peer panel.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    args = parser.parse_args()
    if not args.user_agent:
        parser.error("Provide --user-agent or set SEC_USER_AGENT to a contactable research identifier.")
    root = args.project_root.resolve()
    download_company_facts(
        root / "data/public_comps/company_universe.csv",
        root / "data/public_comps/raw",
        user_agent=args.user_agent,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
