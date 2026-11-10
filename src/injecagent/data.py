"""Download and cache the InjecAgent data files, pinned to an upstream commit.

All assets come from https://github.com/uiuc-kang-lab/InjecAgent (MIT licence)
at a fixed commit, so results are reproducible and the register's "pinned
assets" requirement is met.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict

UPSTREAM_REPO = "uiuc-kang-lab/InjecAgent"
UPSTREAM_COMMIT = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
BASE_URL = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/data"

# sha256 of each file at UPSTREAM_COMMIT; guards against silent upstream changes or corrupt downloads.
FILE_SHA256: Dict[str, str] = {
    "tools.json": "e21a8f70b1d5de4677d6d52642936a322655d79b17a72c84f600550384083a1e",
    "test_cases_dh_base.json": "0a8186468d21389af432e8c7b399ae42264d1b93a07b65c7a489468508604305",
    "test_cases_dh_enhanced.json": "885602716b72c18af80695ce6c2e1f242fa03163bc90b0788b0c5e4ab6216d50",
    "test_cases_ds_base.json": "4daab35c62a3845e8b9400f4dca58b9c9f37e57cd33b2337552557fbb26282e9",
    "test_cases_ds_enhanced.json": "7bc510868df032511053fc40e8470e68a041fb7148d055112093594bf73ab0ce",
    "attacker_simulated_responses.json": "b1da2e1fb75f266c069b832fef6738c295f01d8962fae15bdac9ff6625e5594e",
}

DATA_FILES = (
    "tools.json",
    "test_cases_dh_base.json",
    "test_cases_dh_enhanced.json",
    "test_cases_ds_base.json",
    "test_cases_ds_enhanced.json",
    "attacker_simulated_responses.json",
)


def cache_dir() -> Path:
    root = os.environ.get("INJECAGENT_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "injecagent"
    )
    path = Path(root) / UPSTREAM_COMMIT[:12]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def data_path(name: str) -> Path:
    """Local path of a data file, downloading it on first use."""
    if name not in DATA_FILES:
        raise ValueError(f"unknown InjecAgent data file: {name}")
    local_dir = os.environ.get("INJECAGENT_DATA_DIR")
    if local_dir:  # offline use: point at a checkout of the upstream data/ directory
        return Path(local_dir) / name
    target = cache_dir() / name
    if not target.exists():
        with urllib.request.urlopen(f"{BASE_URL}/{name}", timeout=60) as resp:  # noqa: S310 (fixed https URL)
            payload = resp.read()
        expected = FILE_SHA256.get(name)
        if expected and _sha256(payload) != expected:
            raise RuntimeError(f"checksum mismatch for {name}: expected {expected}, got {_sha256(payload)}")
        tmp = target.with_suffix(".part")
        tmp.write_bytes(payload)
        tmp.replace(target)
    return target


def load_json(name: str) -> Any:
    return json.loads(data_path(name).read_text(encoding="utf-8"))
