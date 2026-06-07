"""ChatGPT-like web chat page for the localhost gateway (dependency-free browser JS).

Talks to the gateway's ``POST /chat`` (reusing AgenticLocal sessions/memory) and
rehydrates history from ``GET /history``. Includes a "Show & copy API" panel that
reveals the OpenAI base_url + token so the user can paste it into VS Code.
"""

from __future__ import annotations

import json


def chat_page_html(*, version: str, api_base: str, token: str | None, model_label: str) -> str:
    cfg = json.dumps({"apiBase": api_base, "token": token or "", "model": model_label})
    # NOTE: token is embedded for this single-user localhost gateway so the page
    # can call /chat and the API panel can show it for VS Code.
    return _TEMPLATE.replace("__CONFIG__", cfg).replace("__VERSION__", version).replace(
        "__MODEL__", _escape(model_label)
    )


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>agentica-core chat</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: #0d1117; color: #e6edf3; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 16px; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .model { font-size: 12px; color: #8b949e; }
  header .spacer { flex: 1; }
  button { background: #238636; color: white; border: 0; border-radius: 6px; padding: 8px 14px;
           font-size: 13px; cursor: pointer; }
  button.secondary { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
  #log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 760px; padding: 10px 14px; border-radius: 10px; white-space: pre-wrap; line-height: 1.45; }
  .user { align-self: flex-end; background: #1f6feb; color: white; }
  .assistant { align-self: flex-start; background: #161b22; border: 1px solid #30363d; }
  .meta { align-self: center; font-size: 11px; color: #8b949e; }
  footer { border-top: 1px solid #30363d; padding: 12px 16px; display: flex; gap: 10px; }
  #input { flex: 1; resize: none; background: #0d1117; color: #e6edf3; border: 1px solid #30363d;
           border-radius: 8px; padding: 10px; font-size: 14px; min-height: 44px; }
  #api { display: none; padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; font-size: 13px; }
  #api code { background: #0d1117; padding: 2px 6px; border-radius: 4px; }
  #api pre { background: #0d1117; padding: 10px; border-radius: 6px; overflow-x: auto; }
  .row { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
</style>
</head>
<body>
<header>
  <h1>agentica-core</h1>
  <span class="model">model: __MODEL__ &middot; v__VERSION__</span>
  <span class="spacer"></span>
  <button class="secondary" id="apiBtn">Show &amp; copy API</button>
  <button class="secondary" id="newBtn">New chat</button>
</header>
<div id="api">
  <div class="row">OpenAI base URL (for VS Code / Continue / Cline):
    <code id="apiBase"></code> <button class="secondary" data-copy="apiBase">Copy</button></div>
  <div class="row">API key: <code id="apiKey"></code> <button class="secondary" data-copy="apiKey">Copy</button></div>
  <div>VS Code (Continue) config snippet:</div>
  <pre id="snippet"></pre>
  <button class="secondary" data-copy="snippet">Copy snippet</button>
</div>
<div id="log"></div>
<footer>
  <textarea id="input" placeholder="Message the agent...  (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send">Send</button>
</footer>
<script>
const CFG = __CONFIG__;
const log = document.getElementById('log');
const input = document.getElementById('input');
let sessionId = localStorage.getItem('sa_session') || null;

function add(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}
function meta(text) {
  const div = document.createElement('div');
  div.className = 'meta'; div.textContent = text; log.appendChild(div);
}
function headers() {
  const h = {'Content-Type': 'application/json'};
  if (CFG.token) h['Authorization'] = 'Bearer ' + CFG.token;
  return h;
}
async function hydrate() {
  if (!sessionId) return;
  try {
    const r = await fetch('/history?session_id=' + encodeURIComponent(sessionId), {headers: headers()});
    if (!r.ok) return;
    const data = await r.json();
    (data.messages || []).forEach(m => {
      if (m.role === 'user' || m.role === 'assistant') add(m.role, m.content);
    });
  } catch (e) {}
}
async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  add('user', text);
  const pending = add('assistant', '…');
  try {
    const r = await fetch('/chat', {method: 'POST', headers: headers(),
      body: JSON.stringify({message: text, session_id: sessionId})});
    const data = await r.json();
    if (data.session_id) { sessionId = data.session_id; localStorage.setItem('sa_session', sessionId); }
    pending.textContent = data.final_answer || data.message || JSON.stringify(data);
  } catch (e) {
    pending.textContent = 'error: ' + e;
  }
}
document.getElementById('send').onclick = send;
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById('newBtn').onclick = () => {
  sessionId = null; localStorage.removeItem('sa_session'); log.innerHTML = ''; meta('new chat');
};
// API panel
const apiBase = document.getElementById('apiBase');
const apiKey = document.getElementById('apiKey');
apiBase.textContent = CFG.apiBase;
apiKey.textContent = CFG.token ? CFG.token : '(none — open access)';
document.getElementById('snippet').textContent = JSON.stringify({
  models: [{ title: 'agentica-core', provider: 'openai', model: CFG.model,
             apiBase: CFG.apiBase, apiKey: CFG.token || 'sk-none' }]
}, null, 2);
document.getElementById('apiBtn').onclick = () => {
  const el = document.getElementById('api');
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
};
document.querySelectorAll('[data-copy]').forEach(b => {
  b.onclick = () => {
    const t = document.getElementById(b.dataset.copy).textContent;
    navigator.clipboard.writeText(t); b.textContent = 'Copied'; setTimeout(() => b.textContent = 'Copy', 1200);
  };
});
hydrate();
</script>
</body>
</html>
"""
