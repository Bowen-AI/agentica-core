"""SSH / SLURM transport: dependency-light wrappers over the system ssh/scp/rsync.

Everything is SYNCHRONOUS (subprocess.run) -- ssh/sbatch calls are blocking and
async would add an event loop for no concurrency benefit in the MVP.

A :class:`Transport` either runs commands on a remote login node (over ssh) or
LOCALLY (``target=None``) -- the local mode makes the whole layer testable without
a cluster (point it at fake ``sbatch``/``squeue`` shims on ``PATH``, or use a real
SSH target of ``localhost`` via :meth:`Transport.ssh_localhost`).
"""

from __future__ import annotations

import getpass
import shlex
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from .config import SSHConfig

# Hardened ssh options: fail fast, no host-key prompts, and multiplex connections
# over a single master (clusters often rate-limit rapid new ssh connections).
SSH_BASE_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/cm-slurm-agentic-%r@%h:%p",
    # Long persistence: status polls + agent tool calls arrive for many minutes;
    # re-dialing through a rate-limiting bastion is what gets connections dropped.
    "-o", "ControlPersist=600s",
]
SSH_TUNNEL_OPTS = [
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
    "-o", "ExitOnForwardFailure=yes",
]


class TransportError(RuntimeError):
    pass


@dataclass
class ExecResult:
    rc: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.rc == 0

    def check(self, what: str = "command") -> "ExecResult":
        if not self.ok:
            raise TransportError(f"{what} failed (rc={self.rc}): {self.err.strip() or self.out.strip()}")
        return self


@dataclass
class Transport:
    """Run commands locally or on a remote login node over ssh."""

    target: SSHConfig | None = None  # None -> run locally
    env: dict[str, str] = field(default_factory=dict)

    # -- factories --
    @classmethod
    def local(cls) -> "Transport":
        return cls(target=None)

    @classmethod
    def ssh_localhost(cls, user: str | None = None) -> "Transport":
        """Real ssh, but to localhost -- exercises the full ssh path with no cluster."""
        return cls(target=SSHConfig(host="127.0.0.1", user=user or getpass.getuser()))

    @classmethod
    def from_cluster(cls, cluster) -> "Transport":
        return cls(target=cluster.ssh)

    # -- ssh argv --
    def _ssh_opts(self) -> list[str]:
        """Connection flags (no host). Only overrides what the YAML set; everything
        else (User, HostName, IdentityFile, ProxyJump, ...) comes from ~/.ssh/config."""
        t = self.target
        assert t is not None
        opts: list[str] = []
        if t.proxy_jump:
            opts += ["-J", t.proxy_jump]
        if t.port:
            opts += ["-p", str(t.port)]
        key = t.expanded_key()
        if key:
            opts += ["-i", key]
        for o in t.options:
            opts += ["-o", o]
        return opts

    def _ssh_prefix(self, extra_opts: Sequence[str] = ()) -> list[str]:
        assert self.target is not None
        return ["ssh", *SSH_BASE_OPTS, *extra_opts, *self._ssh_opts(), self.target.target_str]

    def exec(self, command: str | Sequence[str], timeout: float | None = 120.0) -> ExecResult:
        """Run a shell command (string or argv) locally or on the remote login node."""
        if isinstance(command, (list, tuple)):
            command = " ".join(shlex.quote(str(c)) for c in command)
        if self.target is None:
            argv = ["bash", "-lc", command]
        else:
            argv = [*self._ssh_prefix(), f"bash -lc {shlex.quote(command)}"]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout,
                env=_merged_env(self.env),
            )
        except subprocess.TimeoutExpired as exc:
            raise TransportError(f"command timed out after {timeout}s: {command}") from exc
        except FileNotFoundError as exc:
            raise TransportError(f"executable not found: {exc}") from exc
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    def expand_home(self, path: str) -> str:
        """Resolve a leading ``~`` to the remote/local absolute $HOME, so it is safe
        to bake into PYTHONPATH / scripts where the shell won't expand it."""
        if path.startswith("~"):
            home = self.exec("echo $HOME", timeout=30).out.strip()
            if home:
                return home + path[1:]
        return path

    # -- file staging (rsync; falls back to plain cp locally) --
    def push_dir(self, local_dir: str, remote_dir: str, delete: bool = False) -> ExecResult:
        return self._rsync(local_dir.rstrip("/") + "/", remote_dir, to_remote=True, delete=delete)

    def pull_dir(self, remote_dir: str, local_dir: str, delete: bool = False) -> ExecResult:
        return self._rsync(remote_dir.rstrip("/") + "/", local_dir, to_remote=False, delete=delete)

    def _rsync(self, src: str, dst: str, to_remote: bool, delete: bool) -> ExecResult:
        if self.target is None:
            argv = ["rsync", "-a"]
            if delete:
                argv.append("--delete")
            argv += [src, dst]
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
                if proc.returncode == 127 or "not found" in proc.stderr.lower():
                    return self._tar_sync(src, dst, to_remote)
                return ExecResult(proc.returncode, proc.stdout, proc.stderr)
            except FileNotFoundError:
                return self._tar_sync(src, dst, to_remote)
        ssh_cmd = " ".join(["ssh", *SSH_BASE_OPTS, *self._ssh_opts()])
        remote = self.target.target_str
        argv = ["rsync", "-az", "-e", ssh_cmd]
        if delete:
            argv.append("--delete")
        if to_remote:
            argv += [src, f"{remote}:{dst}"]
        else:
            argv += [f"{remote}:{src}", dst]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
            if proc.returncode == 127 or "not found" in proc.stderr.lower() or "rsync: command not found" in proc.stderr:
                return self._tar_sync(src, dst, to_remote)
            return ExecResult(proc.returncode, proc.stdout, proc.stderr)
        except FileNotFoundError:
            return self._tar_sync(src, dst, to_remote)
        except subprocess.TimeoutExpired as exc:
            raise TransportError("rsync timed out") from exc

    def _tar_sync(self, src: str, dst: str, to_remote: bool) -> ExecResult:
        import os
        src_dir = src.rstrip("/")
        dst_dir = dst.rstrip("/")
        if self.target is None:
            try:
                os.makedirs(dst_dir, exist_ok=True)
                p_local_src = subprocess.Popen(["tar", "-cf", "-", "-C", src_dir, "."], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p_local_dst = subprocess.Popen(["tar", "-xf", "-", "-C", dst_dir], stdin=p_local_src.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p_local_src.stdout.close()
                out_dst, err_dst = p_local_dst.communicate()
                _, err_src = p_local_src.communicate()
                p_local_src.wait()
                rc = p_local_dst.returncode
                err_msg = (err_src.decode("utf-8", "replace") + "\n" + err_dst.decode("utf-8", "replace")).strip()
                return ExecResult(rc, out_dst.decode("utf-8", "replace"), err_msg)
            except Exception as exc:
                return ExecResult(1, "", f"local tar copy failed: {exc}")

        if to_remote:
            remote_cmd = f"mkdir -p {shlex.quote(dst_dir)} && tar -xzf - -C {shlex.quote(dst_dir)}"
            remote_argv = [*self._ssh_prefix(), f"bash -lc {shlex.quote(remote_cmd)}"]
            local_argv = ["tar", "-czf", "-", "-C", src_dir, "."]
            try:
                p_local = subprocess.Popen(local_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p_remote = subprocess.Popen(remote_argv, stdin=p_local.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p_local.stdout.close()
                out_rem, err_rem = p_remote.communicate()
                _, err_loc = p_local.communicate()
                p_local.wait()
                rc = p_remote.returncode
                err_msg = (err_loc.decode("utf-8", "replace") + "\n" + err_rem.decode("utf-8", "replace")).strip()
                return ExecResult(rc, out_rem.decode("utf-8", "replace"), err_msg)
            except Exception as exc:
                return ExecResult(1, "", f"tar push failed: {exc}")
        else:
            remote_cmd = f"tar -czf - -C {shlex.quote(src_dir)} ."
            remote_argv = [*self._ssh_prefix(), f"bash -lc {shlex.quote(remote_cmd)}"]
            local_argv = ["tar", "-xzf", "-", "-C", dst_dir]
            try:
                os.makedirs(dst_dir, exist_ok=True)
                p_remote = subprocess.Popen(remote_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p_local = subprocess.Popen(local_argv, stdin=p_remote.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p_remote.stdout.close()
                out_loc, err_loc = p_local.communicate()
                _, err_rem = p_remote.communicate()
                p_remote.wait()
                rc = p_local.returncode
                err_msg = (err_rem.decode("utf-8", "replace") + "\n" + err_loc.decode("utf-8", "replace")).strip()
                return ExecResult(rc, out_loc.decode("utf-8", "replace"), err_msg)
            except Exception as exc:
                return ExecResult(1, "", f"tar pull failed: {exc}")

    # -- SLURM helpers (run on the login node) --
    def sbatch(self, script_path: str, args: Sequence[str] = ()) -> str:
        """Submit a job script and return its job id (uses --parsable)."""
        cmd = ["sbatch", "--parsable", script_path, *args]
        res = self.exec(cmd).check("sbatch")
        job_id = res.out.strip().splitlines()[0].split(";")[0].strip() if res.out.strip() else ""
        if not job_id:
            raise TransportError(f"sbatch returned no job id: {res.out!r} {res.err!r}")
        return job_id

    def squeue_job(self, job_id: str) -> dict[str, str]:
        """Return {'state':..., 'nodelist':...} for a job, or {} if not in the queue."""
        res = self.exec(["squeue", "-j", str(job_id), "-h", "-o", "%T|%N"])
        line = res.out.strip().splitlines()[0] if res.out.strip() else ""
        if not line:
            return {}
        state, _, nodelist = line.partition("|")
        return {"state": state.strip(), "nodelist": nodelist.strip()}

    def sacct_state(self, job_id: str) -> str:
        res = self.exec(["sacct", "-j", str(job_id), "-n", "-o", "State", "-P"])
        for line in res.out.splitlines():
            s = line.strip()
            if s:
                return s.split()[0]
        return "UNKNOWN"

    def scancel(self, job_id: str) -> ExecResult:
        return self.exec(["scancel", str(job_id)])

    def sinfo(self) -> ExecResult:
        return self.exec(["sinfo", "-o", "%P|%G|%D|%T"], timeout=30)

    def wait_for_node(self, job_id: str, timeout_s: float = 600.0, poll_s: float = 3.0) -> str:
        """Block until the job is RUNNING and return the first node name."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            info = self.squeue_job(job_id)
            state = info.get("state", "")
            node = (info.get("nodelist") or "").split(",")[0].strip()
            if state == "RUNNING" and node and not node.startswith("("):
                return node
            if state in {"FAILED", "CANCELLED", "TIMEOUT", "COMPLETED", ""} and not info:
                # not in queue anymore; check sacct for a terminal state
                final = self.sacct_state(job_id)
                if final not in {"RUNNING", "PENDING", "UNKNOWN"}:
                    raise TransportError(f"job {job_id} ended before running: {final}")
            time.sleep(poll_s)
        raise TransportError(f"timed out waiting for job {job_id} to start")

    # -- SSH port-forward tunnel --
    @contextmanager
    def tunnel(
        self,
        local_port: int,
        remote_host: str,
        remote_port: int,
        readiness_url: str | None = None,
        readiness_timeout_s: float = 60.0,
    ) -> Iterator["Tunnel"]:
        """Open an ``ssh -N -L`` tunnel; optionally poll ``readiness_url`` before yielding."""
        if self.target is None:
            raise TransportError("tunnel requires an ssh target (not local mode)")
        argv = [
            "ssh", *SSH_BASE_OPTS, *SSH_TUNNEL_OPTS, "-N", *self._ssh_opts(),
            "-L", f"{local_port}:{remote_host}:{remote_port}", self.target.target_str,
        ]
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tun = Tunnel(local_port=local_port, remote=f"{remote_host}:{remote_port}", proc=proc)
        try:
            if readiness_url:
                _wait_http(readiness_url, readiness_timeout_s, proc)
            else:
                _wait_port(local_port, 15.0, proc)
            yield tun
        finally:
            tun.close()


@dataclass
class Tunnel:
    local_port: int
    remote: str
    proc: subprocess.Popen

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _merged_env(extra: dict[str, str]) -> dict[str, str] | None:
    if not extra:
        return None
    import os

    env = dict(os.environ)
    env.update(extra)
    return env


def _wait_port(port: int, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            raise TransportError(f"ssh tunnel exited early: {err.strip()}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.3)
    raise TransportError(f"tunnel local port {port} never opened")


def _wait_http(url: str, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            raise TransportError(f"ssh tunnel exited early: {err.strip()}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001 - retry until ready
            last = str(exc)
        time.sleep(0.5)
    raise TransportError(f"endpoint not ready at {url} within {timeout_s}s ({last})")
