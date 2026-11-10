import pytest

from agentica_core import doctor, voice_provision


@pytest.mark.parametrize("installed", [True, False])
def test_headless_mac_checks_mlx_presence_without_importing_it(monkeypatch, installed):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(voice_provision, "_metal_available", lambda: False)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda _: object() if installed else None)
    imported = []
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: imported.append(name))
    result = doctor.check(require_voice=True)
    assert "mlx_whisper" not in imported
    assert "faster_whisper" in imported
    assert result["ok"] is installed
    if not installed:
        assert "mlx_whisper: package is missing" in result["errors"]
