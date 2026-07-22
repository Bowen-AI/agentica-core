"""Transport: local exec + fake-SLURM shims + (best-effort) ssh-localhost."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from agentica_core.transport import Transport, TransportError


def _make_fake_slurm(bindir: Path) -> dict[str, str]:
    bindir.mkdir(parents=True, exist_ok=True)
    shims = {
        "sbatch": "#!/bin/bash\necho 990017\n",                       # --parsable job id
        "squeue": "#!/bin/bash\necho 'RUNNING|gpu-node-07'\n",        # %T|%N
        "scancel": "#!/bin/bash\nexit 0\n",
        "sacct": "#!/bin/bash\necho COMPLETED\n",
    }
    for name, body in shims.items():
        p = bindir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return {"PATH": f"{bindir}:{os.environ['PATH']}"}


def test_local_exec():
    t = Transport.local()
    res = t.exec("echo hello")
    assert res.ok
    assert res.out.strip() == "hello"


def test_fake_slurm_submit_and_resolve(tmp_path):
    env = _make_fake_slurm(tmp_path / "bin")
    t = Transport(target=None, env=env)

    job_id = t.sbatch("/tmp/whatever.sbatch")
    assert job_id == "990017"

    info = t.squeue_job(job_id)
    assert info == {"state": "RUNNING", "nodelist": "gpu-node-07"}

    node = t.wait_for_node(job_id, timeout_s=5, poll_s=0.1)
    assert node == "gpu-node-07"

    assert t.scancel(job_id).ok


def test_sbatch_no_jobid_raises(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    p = bindir / "sbatch"
    p.write_text("#!/bin/bash\nexit 0\n")  # prints nothing
    p.chmod(0o755)
    t = Transport(target=None, env={"PATH": f"{bindir}:{os.environ['PATH']}"})
    with pytest.raises(TransportError):
        t.sbatch("/tmp/x.sbatch")


def _ssh_localhost_works() -> bool:
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "true"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _ssh_localhost_works(), reason="passwordless ssh localhost not available")
def test_ssh_localhost_exec():
    t = Transport.ssh_localhost()
    res = t.exec("echo via-ssh")
    assert res.ok and res.out.strip() == "via-ssh"


def test_local_tar_sync(tmp_path):
    t = Transport.local()
    src = tmp_path / "src"
    src.mkdir()
    (src / "file1.txt").write_text("content1")
    (src / "sub").mkdir()
    (src / "sub" / "file2.txt").write_text("content2")

    dst = tmp_path / "dst"
    res = t._tar_sync(str(src), str(dst), to_remote=True)
    assert res.ok
    assert (dst / "file1.txt").read_text() == "content1"
    assert (dst / "sub" / "file2.txt").read_text() == "content2"


def test_rsync_fallback_to_tar(tmp_path, monkeypatch):
    t = Transport.local()
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    dst = tmp_path / "dst"

    # Mock subprocess.run to raise FileNotFoundError when calling rsync
    import subprocess
    orig_run = subprocess.run
    def mock_run(argv, *args, **kwargs):
        if argv[0] == "rsync":
            raise FileNotFoundError("rsync not found")
        return orig_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", mock_run)

    res = t.push_dir(str(src), str(dst))
    assert res.ok
    assert (dst / "a.txt").read_text() == "hello"
