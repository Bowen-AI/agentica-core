#!/usr/bin/env bash
# Bring up Agentica's expressive remote TTS on a GPU box and tunnel it back.
#
#   scripts/tts-up.sh <ssh-host> [engine] [port]
#   scripts/tts-up.sh pinotage.usc.edu chatterbox 8780
#
# One-time on the box (creates a 'agxtts' conda env):
#   ssh <host> 'source ~/miniconda3/etc/profile.d/conda.sh; conda create -y -n agxtts python=3.10; \
#               conda activate agxtts; pip install chatterbox-tts'   # or dia / TTS (xtts)
#
# This script copies the server, starts it detached on the box, opens an SSH
# tunnel (local:PORT -> box:PORT), and prints the AGENTICA_TTS_URL to export so
# the voice gateway routes TTS through it (falls back to local Kokoro if it's down).
set -euo pipefail

HOST="${1:?usage: tts-up.sh <ssh-host> [engine] [port]}"
ENGINE="${2:-chatterbox}"
PORT="${3:-8780}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "→ copying tts_server.py to ${HOST}…"
scp -q "${HERE}/agentica_core/tts_server.py" "${HOST}:~/agentica_tts_server.py"

echo "→ starting '${ENGINE}' TTS server on ${HOST}:${PORT} (detached)…"
ssh "${HOST}" "bash -lc '
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate agxtts
  pkill -f agentica_tts_server.py || true
  nohup python ~/agentica_tts_server.py --engine ${ENGINE} --port ${PORT} > ~/tts_server.log 2>&1 &
  sleep 1; echo started
'"

echo "→ waiting for the model to load on ${HOST} (first run downloads weights)…"
for i in $(seq 1 60); do
  if ssh "${HOST}" "curl -s --max-time 3 http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
    ssh "${HOST}" "curl -s http://127.0.0.1:${PORT}/health"; echo; break
  fi
  sleep 5
done

echo "→ opening SSH tunnel localhost:${PORT} -> ${HOST}:${PORT} (leave this running)…"
echo "   then in another shell:  export AGENTICA_TTS_URL=http://127.0.0.1:${PORT}"
echo "   and (re)start the backend so the voice uses the expressive engine."
exec ssh -N -L "${PORT}:127.0.0.1:${PORT}" "${HOST}"
