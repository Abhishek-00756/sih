"""Evidence anchoring abstraction.

The local anchor is the immediately usable cybersecurity layer: it stores a
SHA-256 hash chain containing incident metadata and the evidence hash. The
interface is intentionally small so a Hyperledger Fabric adapter can replace
or augment it later without changing incident-generation code.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict


class EvidenceLedger:
    def __init__(self, db_path: Path | str = "border_evidence_ledger.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_blocks (
                    block_index INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    previous_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    block_hash TEXT NOT NULL UNIQUE
                )
                """
            )
            conn.commit()

    @staticmethod
    def _block_hash(previous_hash: str, payload_json: str) -> str:
        raw = f"{previous_hash}|{payload_json}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def anchor(self, evidence: Dict[str, Any]) -> dict:
        payload_json = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT block_index, block_hash FROM evidence_blocks ORDER BY block_index DESC LIMIT 1"
            ).fetchone()
            previous_hash = row[1] if row else "0" * 64
            created_at = time.time()
            block_hash = self._block_hash(previous_hash, payload_json)
            cur = conn.execute(
                "INSERT INTO evidence_blocks(created_at, previous_hash, payload_json, block_hash) VALUES (?, ?, ?, ?)",
                (created_at, previous_hash, payload_json, block_hash),
            )
            conn.commit()
            return {
                "backend": "local_sha256_chain",
                "block_index": int(cur.lastrowid),
                "block_hash": block_hash,
                "previous_hash": previous_hash,
                "anchored_at": created_at,
            }

    def verify(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT block_index, previous_hash, payload_json, block_hash FROM evidence_blocks ORDER BY block_index"
            ).fetchall()

        previous = "0" * 64
        for block_index, stored_previous, payload_json, stored_hash in rows:
            if stored_previous != previous:
                return {"valid": False, "failed_block": block_index, "reason": "previous hash mismatch"}
            expected = self._block_hash(previous, payload_json)
            if expected != stored_hash:
                return {"valid": False, "failed_block": block_index, "reason": "block hash mismatch"}
            previous = stored_hash
        return {"valid": True, "blocks": len(rows), "head_hash": previous}
