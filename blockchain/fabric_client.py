"""Small Fabric client used by BORDER SENTINEL.

The dashboard runs on the operator workstation alongside the Fabric test

network during development. The client intentionally uses the official

Fabric peer CLI so the Python application does not depend on the legacy

fabric-sdk-py package. It invokes the deployed evidence chaincode through

both Org1 and Org2 peers and waits for commit events.

"""

from __future__ import annotations

import json

import os

import re

import subprocess

import threading

from pathlib import Path

from typing import Any, Dict, Optional


TXID_RE = re.compile(r"txid \\[([0-9a-fA-F]{64})\\]")


class FabricEvidenceClient:

    def __init__(

        self,

        network_path: Optional[str | Path] = None,

        channel: Optional[str] = None,

        chaincode: Optional[str] = None,

        timeout_seconds: float = 20.0,

    ) -> None:

        self.network_path = Path(

            network_path

            or os.environ.get("FABRIC_TEST_NETWORK", "~/fabric-samples/test-network")

        ).expanduser().resolve()

        self.channel = channel or os.environ.get("FABRIC_CHANNEL", "evidencechannel")

        self.chaincode = chaincode or os.environ.get("FABRIC_CHAINCODE", "evidence")

        self.timeout_seconds = float(

            os.environ.get("FABRIC_INVOKE_TIMEOUT", str(timeout_seconds))

        )

        self.enabled = os.environ.get(

            "BORDER_SENTINEL_BLOCKCHAIN_ENABLED", "true"

        ).strip().lower() not in {"0", "false", "no", "off"}

        self.peer_bin = (self.network_path.parent / "bin" / "peer").resolve()

        base = self.network_path / "organizations"

        self.orderer_ca = (

            base / "ordererOrganizations/example.com/orderers/orderer.example.com/"

            "msp/tlscacerts/tlsca.example.com-cert.pem"

        )

        self.org1_peer_ca = (

            base / "peerOrganizations/org1.example.com/peers/"

            "peer0.org1.example.com/tls/ca.crt"

        )

        self.org2_peer_ca = (

            base / "peerOrganizations/org2.example.com/peers/"

            "peer0.org2.example.com/tls/ca.crt"

        )

        self.org1_msp = (

            base / "peerOrganizations/org1.example.com/users/"

            "Admin@org1.example.com/msp"

        )

        self.config_path = self.network_path.parent / "config"

        self._lock = threading.Lock()

        self._last_result: Dict[str, Any] = {

            "status": "DISABLED" if not self.enabled else "READY",

            "transaction_id": None,

            "error": None,

        }


    def _configured(self) -> bool:

        required = (

            self.peer_bin,

            self.orderer_ca,

            self.org1_peer_ca,

            self.org2_peer_ca,

            self.org1_msp,

            self.config_path,

        )

        return all(path.exists() for path in required)


    def _env(self) -> dict[str, str]:

        env = os.environ.copy()

        bin_dir = str(self.peer_bin.parent)

        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

        env["FABRIC_CFG_PATH"] = str(self.config_path)

        env["CORE_PEER_TLS_ENABLED"] = "true"

        env["CORE_PEER_LOCALMSPID"] = "Org1MSP"

        env["CORE_PEER_TLS_ROOTCERT_FILE"] = str(self.org1_peer_ca)

        env["CORE_PEER_MSPCONFIGPATH"] = str(self.org1_msp)

        env["CORE_PEER_ADDRESS"] = "localhost:7051"

        return env


    def status(self) -> dict:

        with self._lock:

            result = dict(self._last_result)

        return {

            "enabled": self.enabled,

            "configured": self._configured(),

            "status": result["status"],

            "transaction_id": result.get("transaction_id"),

            "error": result.get("error"),

            "channel": self.channel,

            "chaincode": self.chaincode,

        }


    def record_evidence(

        self,

        event_id: str,

        camera_id: str,

        global_id: int,

        event_type: str,

        timestamp: str,

        direction: str,

        evidence_sha256: str,

        previous_hash: str,

    ) -> dict:

        if not self.enabled:

            return {"status": "DISABLED", "transaction_id": None, "error": None}

        if not self._configured():

            result = {

                "status": "UNAVAILABLE",

                "transaction_id": None,

                "error": "Fabric test-network paths are not available",

            }

            with self._lock:

                self._last_result = result

            return result

        ctor = json.dumps(

            {

                "function": "RecordEvidence",

                "Args": [

                    str(event_id),

                    str(camera_id),

                    str(int(global_id)),

                    str(event_type),

                    str(timestamp),

                    str(direction or ""),

                    str(evidence_sha256),

                    str(previous_hash or ""),

                ],

            },

            separators=(",", ":"),

        )

        command = [

            str(self.peer_bin), "chaincode", "invoke",

            "-o", "localhost:7050",

            "--ordererTLSHostnameOverride", "orderer.example.com",

            "--tls", "--cafile", str(self.orderer_ca),

            "-C", self.channel, "-n", self.chaincode,

            "--peerAddresses", "localhost:7051",

            "--tlsRootCertFiles", str(self.org1_peer_ca),

            "--peerAddresses", "localhost:9051",

            "--tlsRootCertFiles", str(self.org2_peer_ca),

            "--waitForEvent",

            "--waitForEventTimeout", f"{max(5, int(self.timeout_seconds))}s",

            "-c", ctor,

        ]

        try:

            completed = subprocess.run(

                command,

                cwd=self.network_path,

                env=self._env(),

                capture_output=True,

                text=True,

                timeout=max(10.0, self.timeout_seconds + 5.0),

                check=False,

            )

        except (OSError, subprocess.SubprocessError) as exc:

            result = {"status": "FAILED", "transaction_id": None, "error": str(exc)}

            with self._lock:

                self._last_result = result

            return result

        output = (completed.stdout or "") + "\n" + (completed.stderr or "")

        match = TXID_RE.search(output)

        transaction_id = match.group(1) if match else None

        success = completed.returncode == 0 and "Chaincode invoke successful" in output

        if success:

            result = {

                "status": "ANCHORED",

                "transaction_id": transaction_id,

                "error": None,

            }

        else:

            result = {

                "status": "FAILED",

                "transaction_id": transaction_id,

                "error": output.strip()[-1200:] or f"peer exited with code {completed.returncode}",

            }

        with self._lock:

            self._last_result = result

        return result

