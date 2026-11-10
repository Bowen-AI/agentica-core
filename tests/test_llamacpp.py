"""Managed llama.cpp lifecycle without downloading multi-GB model weights."""
from contextlib import contextmanager
import io
import threading
from types import SimpleNamespace

import pytest

from agentica_core import apiserver, llamacpp


@pytest.fixture
def models(tmp_path):
    paths = [tmp_path / 'models with spaces' / 'one.gguf', tmp_path / 'two.gguf']
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'GGUFtest')
    return paths


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    processes = []
    monkeypatch.setattr(llamacpp, "free_port", lambda: 20000 + len(processes))
    class Process:
        def __init__(self, args, **kwargs):
            self.args, self.kwargs, self.stopped = args, kwargs, False
            processes.append(self)
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, **kwargs): return 0
        def kill(self): self.stopped = True
    monkeypatch.setattr(llamacpp, 'find_server', lambda: '/fake/llama-server')
    monkeypatch.setattr(llamacpp.subprocess, 'Popen', Process)
    monkeypatch.setattr(llamacpp.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'{"status":"ok"}'))
    server = llamacpp.LlamaServer(tmp_path)
    yield server, processes
    server.close()


def test_reuses_process_switches_model_and_cleans_up(runtime, models):
    server, processes = runtime
    with server.use(str(models[0])) as first:
        assert first.startswith('http://127.0.0.1:')
    with server.use(str(models[0])) as second:
        assert second == first
    assert len(processes) == 1
    assert processes[0].args[processes[0].args.index('--model')+1] == str(models[0])
    assert '--jinja' in processes[0].args
    assert processes[0].kwargs.get('shell', False) is False
    with server.use(str(models[1])):
        assert processes[0].stopped
    assert len(processes) == 2
    server.close()
    assert processes[1].stopped


def test_model_switch_waits_for_active_lease(runtime, models):
    server, processes = runtime
    requested, acquired = threading.Event(), threading.Event()
    def switch():
        requested.set()
        with server.use(str(models[1])):
            acquired.set()
    with server.use(str(models[0])):
        thread = threading.Thread(target=switch)
        thread.start()
        assert requested.wait(1)
        assert not acquired.wait(.05)
        assert not processes[0].stopped
    thread.join(timeout=2)
    assert acquired.is_set()


def test_missing_binary_and_invalid_file_are_actionable(monkeypatch, tmp_path, models):
    server = llamacpp.LlamaServer(tmp_path)
    monkeypatch.setattr(llamacpp, 'find_server', lambda: None)
    with pytest.raises(RuntimeError, match='brew install llama.cpp'):
        with server.use(str(models[0])): pass
    with pytest.raises(ValueError, match='absolute path'):
        llamacpp.gguf_path('relative.gguf')
    with pytest.raises(ValueError, match='not found'):
        llamacpp.gguf_path(str(tmp_path/'missing.gguf'))
    models[0].write_bytes(b'wrong')
    with pytest.raises(ValueError, match='Not a GGUF'):
        llamacpp.gguf_path(str(models[0]))


def test_startup_timeout_cleans_up_and_can_retry(runtime, models, monkeypatch):
    server, processes = runtime
    server.timeout = .01
    monkeypatch.setattr(llamacpp.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'{"status":"loading"}'))
    with pytest.raises(RuntimeError, match='did not become ready'):
        with server.use(str(models[0])): pass
    assert processes[0].stopped
    assert server._log is None
    monkeypatch.setattr(llamacpp.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'{"status":"ok"}'))
    with server.use(str(models[0])): pass
    assert len(processes) == 2


def test_crashed_server_is_restarted(runtime, models):
    server, processes = runtime
    with server.use(str(models[0])): pass
    processes[0].stopped = True
    with server.use(str(models[0])): pass
    assert len(processes) == 2


def test_start_failure_closes_log(runtime, models, monkeypatch):
    server, _ = runtime
    def fail(*a, **k): raise OSError('not executable')
    monkeypatch.setattr(llamacpp.subprocess, 'Popen', fail)
    with pytest.raises(OSError):
        with server.use(str(models[0])): pass
    assert server._log is None


def test_explicit_binary_does_not_fall_back(monkeypatch):
    monkeypatch.setenv('AGENTICA_LLAMA_SERVER', '/missing/server')
    monkeypatch.setattr(llamacpp.shutil, 'which', lambda _: '/some/other/server')
    assert llamacpp.find_server() is None


@pytest.mark.parametrize('workspace_target', ['local', 'remote-tools'])
def test_gguf_uses_agent_loop_with_independent_workspace(runtime, models, tmp_path, monkeypatch, workspace_target):
    server, _ = runtime
    calls = []
    class App:
        def chat(self, message, session_id, **kwargs):
            calls.append(('chat', message, kwargs))
            return {'final_answer': 'agent did it', 'steps': [{'tool_name': 'read_file'}]}
    def build(**kwargs):
        calls.append(('build', kwargs))
        return App()
    monkeypatch.setattr(apiserver.gateway, 'build_app', build)
    state = apiserver.State(ollama_host='http://unused', model='unused', workspace=str(tmp_path), db_path=str(tmp_path/'state.db'))
    state.llama = server
    app = state.app_for(str(tmp_path), target='local', model=str(models[0]), engine='llamacpp', workspace_target=workspace_target)
    # Switch between app creation and turn execution: the lease restores the right model.
    with server.use(str(models[1])): pass
    result = app.chat('read the file', None, approval_callback=lambda *_: False)
    assert result['steps'][0]['tool_name'] == 'read_file'
    builds = [call[1] for call in calls if call[0] == 'build']
    assert all(b['provider'] == 'openai-compatible' and b['tools_target'] == workspace_target for b in builds)
    assert builds[-1]['model_name'] == str(models[0])
    assert calls[-1][0] == 'chat'
    with pytest.raises(ValueError, match='local GGUF'):
        state.runtime_for('remote', str(models[0]), 'llamacpp')
    with pytest.raises(ValueError, match='Unknown model engine'):
        state.runtime_for('local', str(models[0]), 'typo')


def test_native_tool_history_and_system_messages_fit_gguf_templates(monkeypatch):
    from agentica_core.llama_model import LlamaChatModel
    from agentic_loop.types import Message, ModelResponse
    model = LlamaChatModel('test.gguf', 'http://127.0.0.1:1234/v1')
    monkeypatch.setattr('agentic_loop.providers.openai_compatible.OpenAICompatibleChatModel.respond',
                        lambda *a: ModelResponse.call('read_file', {'path': 'input.txt'}, 'call1'))
    model.respond([], [], '')
    rendered = model._messages([
        Message('system', 'Follow workspace policy.'), Message('user', 'read the file'),
        Message('system', 'Runtime context.'),
        Message('assistant', 'Tool call requested: read_file'),
        Message('tool', 'random file contents', name='read_file', tool_call_id='call1'),
    ])
    assert rendered[0] == {'role': 'system', 'content': 'Follow workspace policy.\n\nRuntime context.'}
    assert [m['role'] for m in rendered] == ['system','user','assistant','tool']
    assert rendered[2]['tool_calls'][0]['function'] == {'name':'read_file','arguments':'{"path": "input.txt"}'}
    assert rendered[3]['tool_call_id'] == 'call1'
    assert rendered[3]['content'] == 'random file contents'
    restored = LlamaChatModel('test.gguf', 'http://127.0.0.1:1234/v1')
    assert restored._messages([Message('tool', 'saved result', name='read_file')]) == [
        {'role':'user','content':'Previous tool result (read_file):\nsaved result'}]


def test_local_gguf_job_uses_native_adapter_and_persists_outcome(runtime, models, tmp_path, monkeypatch):
    import time
    from agentica_core.llama_model import LlamaChatModel
    server, _ = runtime
    state = apiserver.State(ollama_host='http://unused', model='unused', workspace=str(tmp_path), db_path=str(tmp_path/'job.db'))
    state.llama = server
    calls = []
    def run(plan, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(passed=True, to_dict=lambda: {'passed':True})
    monkeypatch.setattr(apiserver, 'run_job', run)
    result = apiserver.submit_local(state, {'goal':'test','lines':[]}, str(tmp_path), str(models[0]), 'llamacpp')
    job = state.local_jobs[result['local_id']]
    for _ in range(100):
        if job['status'] != 'running': break
        time.sleep(.01)
    assert job['status'] == 'passed', job
    assert calls[0]['provider'] == 'openai-compatible'
    assert isinstance(calls[0]['model'], LlamaChatModel)
    assert apiserver._load_local_jobs(state.db_path)[result['local_id']]['status'] == 'passed'
