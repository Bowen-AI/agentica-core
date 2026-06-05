"""Targeting any ~/.ssh/config host: alias-or-yaml resolve, ssh-opts, host listing."""

from pathlib import Path

from agentica_core import sshconfig
from agentica_core.config import ClusterConfig, SSHConfig
from agentica_core.transport import Transport

EX = Path(__file__).resolve().parent.parent / "examples"


def test_resolve_yaml_vs_alias():
    # existing file -> load
    c1 = ClusterConfig.resolve(EX / "cluster.yaml")
    assert c1.ssh.host == "login.cluster.edu"
    # non-file -> treat as ssh alias, inherit everything else from ~/.ssh/config
    c2 = ClusterConfig.resolve("some-random-alias")
    assert c2.ssh.host == "some-random-alias"
    assert c2.ssh.user is None  # comes from ~/.ssh/config at connect time


def test_ssh_target_str_optional_user():
    assert SSHConfig(host="h").target_str == "h"
    assert SSHConfig(host="h", user="u").target_str == "u@h"


def test_ssh_opts_honor_config():
    t = Transport(target=SSHConfig(host="alias", user="me", port=2222,
                                   key_path="/tmp/k", proxy_jump="bastion",
                                   options=["StrictHostKeyChecking=no"]))
    argv = t._ssh_prefix()
    assert "-J" in argv and "bastion" in argv
    assert "-p" in argv and "2222" in argv
    assert "-i" in argv and "/tmp/k" in argv
    assert argv[-1] == "me@alias"
    # extra -o option threaded through
    assert "StrictHostKeyChecking=no" in argv


def test_ssh_opts_minimal_when_unset():
    # bare alias: no -p/-i/-J, let ~/.ssh/config drive it
    t = Transport(target=SSHConfig(host="alias"))
    argv = t._ssh_prefix()
    assert "-p" not in argv and "-i" not in argv and "-J" not in argv
    assert argv[-1] == "alias"


def test_list_hosts_parser(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    cfg.write_text(
        "Host gpu-box\n  HostName 10.0.0.5\n  User alice\n\n"
        "Host cluster\n  HostName login.hpc.edu\n  User bob\n  ProxyJump bastion\n\n"
        "Host *.internal\n  User wild\n",  # wildcard -> skipped
        encoding="utf-8",
    )
    monkeypatch.setattr(sshconfig, "ssh_config_path", lambda: cfg)
    hosts = {h["alias"]: h for h in sshconfig.list_hosts()}
    assert set(hosts) == {"gpu-box", "cluster"}  # wildcard excluded
    assert hosts["gpu-box"]["hostname"] == "10.0.0.5" and hosts["gpu-box"]["user"] == "alice"
    assert hosts["cluster"]["proxy_jump"] == "bastion"


def test_scheduler_field_default():
    c = ClusterConfig.resolve("plain-box")
    assert c.scheduler == "auto"
