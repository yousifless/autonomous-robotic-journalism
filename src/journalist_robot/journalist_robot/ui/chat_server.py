"""Flask server: shows the latest articles and answers questions about the footage.

Binds to 127.0.0.1 by default. Expose it on other interfaces only on a
trusted network; it has no authentication.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List

MAX_QUESTION_CHARS = 1000

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Autonomous Journalist</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f5f5f4;color:#1c1917}
 main{max-width:1100px;margin:24px auto;padding:0 16px;display:grid;grid-template-columns:2fr 1fr;gap:16px}
 @media(max-width:800px){main{grid-template-columns:1fr}}
 section{background:#fff;border:1px solid #e7e5e4;border-radius:10px;padding:16px}
 h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:0 0 8px;color:#57534e}
 .lead{font-weight:600} .meta{font-size:12px;color:#78716c;margin-bottom:12px}
 article{white-space:pre-wrap;line-height:1.55;margin-bottom:24px}
 #log{height:360px;overflow-y:auto;font-size:14px;border-top:1px solid #e7e5e4;padding-top:8px}
 .row{display:flex;gap:8px;margin-top:8px} input{flex:1;padding:8px;border:1px solid #d6d3d1;border-radius:6px}
 button{padding:8px 14px;border:0;border-radius:6px;background:#1c1917;color:#fff;cursor:pointer}
</style></head><body><main>
<section><h2>Articles</h2><div id="articles">Loading...</div></section>
<section><h2>Ask about the footage</h2><div id="log"></div>
<div class="row"><input id="q" placeholder="What happened at the start?"><button id="ask">Ask</button></div></section>
</main>
<script>
function el(tag, cls, text){const e=document.createElement(tag); if(cls) e.className=cls; if(text!==undefined) e.textContent=text; return e;}
async function load(){
  const data = await (await fetch('/articles')).json();
  const box = document.getElementById('articles'); box.replaceChildren();
  if(!data.length){ box.textContent='No article yet.'; return; }
  for(const a of data){
    const art = el('article');
    art.append(el('h1', '', a.headline));
    art.append(el('div', 'meta', `Segment ${a.segment_id} - internal score ${a.acceptance_score.toFixed(2)} (${a.accepted ? 'accepted' : 'not accepted'}) - not independently verified`));
    art.append(el('p', 'lead', a.lead));
    art.append(el('div', '', a.body));
    box.append(art);
  }
}
async function ask(){
  const input = document.getElementById('q'); const q = input.value.trim(); if(!q) return; input.value='';
  const log = document.getElementById('log');
  log.append(el('p', '', 'Q: ' + q));
  try{
    const r = await fetch('/query', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query:q})});
    log.append(el('p', '', 'A: ' + (await r.json()).answer));
  }catch(e){ log.append(el('p', '', 'Error: ' + e)); }
  log.scrollTop = log.scrollHeight;
}
document.getElementById('ask').onclick = ask;
document.getElementById('q').addEventListener('keydown', e => { if(e.key==='Enter') ask(); });
load(); setInterval(load, 10000);
</script></body></html>"""


class ChatServer:
    """``answer_fn(question) -> str`` is supplied by the journalist node."""

    def __init__(self, answer_fn: Callable[[str], str], host: str = "127.0.0.1", port: int = 8890) -> None:
        self.answer_fn = answer_fn
        self.host = host
        self.port = port
        self._articles: List[Dict] = []
        self._lock = threading.Lock()

    def publish_article(self, article: Dict) -> None:
        with self._lock:
            self._articles.insert(0, article)

    def start(self) -> None:
        from flask import Flask, jsonify, request

        app = Flask("journalist_robot_ui")
        allowed_hosts = {f"{h}:{self.port}" for h in ("127.0.0.1", "localhost", "[::1]", self.host)}

        @app.before_request
        def check_host():
            # Rejects DNS-rebinding requests (a web page whose domain points at 127.0.0.1).
            if self.host in ("127.0.0.1", "localhost", "::1") and request.host not in allowed_hosts:
                return jsonify({"error": "forbidden host"}), 403
            return None

        @app.get("/")
        def index():
            return PAGE

        @app.get("/articles")
        def articles():
            with self._lock:
                return jsonify(list(self._articles))

        @app.post("/query")
        def query():
            q = str((request.get_json(silent=True) or {}).get("query", "")).strip()
            if not q:
                return jsonify({"answer": "Empty question."})
            if len(q) > MAX_QUESTION_CHARS:
                return jsonify({"answer": f"Please keep questions under {MAX_QUESTION_CHARS} characters."})
            try:
                return jsonify({"answer": self.answer_fn(q)})
            except Exception:  # never crash the server or leak internals to the client
                logging.getLogger(__name__).exception("Query failed")
                return jsonify({"answer": "The question could not be answered (see the node log)."})

        threading.Thread(target=lambda: app.run(host=self.host, port=self.port, debug=False,
                                                use_reloader=False), daemon=True).start()
