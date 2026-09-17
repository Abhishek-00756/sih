from dashboard_pairing import PairingManager


def test_pairing_payload_is_short_lived_and_unique():
    manager = PairingManager(ttl_seconds=300)
    a = manager.create("CAM-01", "https://192.168.1.10:8443")
    b = manager.create("CAM-02", "https://192.168.1.10:8443")
    assert a.token != b.token
    assert manager.get(a.token) is a
    assert a.payload["camera_id"] == "CAM-01"
    assert a.payload["type"] == "border-sentinel-camera-pair"


def test_pairing_payload_encodes_without_raw_token_url_unsafe_chars():
    manager = PairingManager()
    session = manager.create("CAM-03", "https://192.168.1.10:8443")
    encoded = manager.encode_payload(session.payload)
    assert encoded
    assert " " not in encoded
    assert manager.fingerprint(session.token)
