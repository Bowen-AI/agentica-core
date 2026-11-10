"""Opt-in real-model smoke: prove the selected engine uses workspace tools.

python scripts/inference_smoke.py --engine llamacpp --model /absolute/model.gguf
AGENTICA_LLAMA_SERVER may point to an unpacked llama.cpp release binary.
All agent file changes are confined to a disposable temporary workspace.
"""
import argparse
import json
import secrets
import tempfile
from pathlib import Path

from agentica_core.apiserver import State


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['ollama', 'llamacpp'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--ollama-host', default='http://127.0.0.1:11434')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='agentica-inference-') as folder:
        workspace = Path(folder)
        marker = secrets.token_hex(12)
        (workspace/'input.txt').write_text(marker)
        state = State(ollama_host=args.ollama_host, model=args.model, workspace=folder,
                      db_path=str(workspace/'agent.db'))
        try:
            app = state.app_for(folder, model=args.model, engine=args.engine)
            result = app.chat(f'Use read_file with path {workspace / "input.txt"}. Then use write_file with path {workspace / "output.txt"} '
                              'containing exactly the contents of input.txt. Do not use shell commands. '
                              'After the write succeeds, answer DONE.', max_steps=12)
            steps = [step.get('tool_name') for step in result.get('steps', [])]
            output = workspace/'output.txt'
            passed = output.is_file() and output.read_text().strip() == marker
            report = {'engine': args.engine, 'model': args.model, 'passed': passed,
                      'tools': steps, 'answer': result.get('final_answer')}
            print(json.dumps(report, indent=2))
            if not passed:
                raise RuntimeError('Real-model tool-use check failed: output.txt did not match input.txt')
        finally:
            state.llama.close()


if __name__ == '__main__':
    main()
