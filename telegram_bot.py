"""Chatty in Telegram: answers allowed chats with the notebook's model.

Run: uv run --env-file .env python telegram_bot.py
The first run trains the model (~10 min) and saves it to model/; later runs load it.
"""

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

os.environ.setdefault("MPLBACKEND", "Agg")  # the notebook's plots must not open windows

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = {int(i) for i in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if i.strip()}
API = f"https://api.telegram.org/bot{TOKEN}/"
# python.org builds on macOS ship without CA certificates; certifi brings them
TLS = ssl.create_default_context(cafile=certifi.where())


def load_notebook() -> dict:
    """Run the notebook's own code up to the inference models, so there is one copy of the model code."""
    cells = ["".join(c["source"]) for c in json.loads(Path("chatty_bot.ipynb").read_text())["cells"]
             if c["cell_type"] == "code"]
    ns: dict = {}
    for src in cells:
        if "model.fit(" in src and Path("model/chatty.keras").exists():
            ns["model"].load_weights("model/chatty.keras")
            continue
        exec(src, ns)
        if "model.fit(" in src:
            exec(next(c for c in cells if "model.save(" in c), ns)
        if "decoder_model =" in src:
            return ns
    raise RuntimeError("notebook has no cell defining decoder_model")


def call(method: str, **params) -> dict:
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(API + method, data, timeout=60, context=TLS) as resp:
        return json.load(resp)


def main():
    nb = load_notebook()
    histories: dict[int, list] = {}  # chat id -> recent exchanges; the model only sees the last MEMORY
    print("chatty is listening; allowed chats:", ALLOWED or "none", flush=True)
    offset = 0
    while True:
        try:
            updates = call("getUpdates", offset=offset, timeout=50)["result"]
        except OSError as e:  # network hiccup: retry instead of crashing
            print("telegram:", e)
            time.sleep(5)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message") or {}
            chat, text = msg.get("chat", {}).get("id"), msg.get("text")
            if not chat or not text:
                continue
            print(f"{chat}: {text!r}", flush=True)
            if chat not in ALLOWED:
                print(f"ignored chat {chat} ({msg['chat'].get('first_name') or msg['chat'].get('title')});"
                      " add it to ALLOWED_CHAT_IDS to let it talk", flush=True)
                continue
            history = histories.setdefault(chat, [])
            text = text.replace("!", "")  # people write "привет!"; the training data never has "!"
            reply = nb["detokenize"](nb["generate"](nb["context_tokens"](history, text))[0]) or "..."
            history.append({"user": text, "bot": reply})
            del history[:-nb["MEMORY"]]
            call("sendMessage", chat_id=chat, text=reply)


if __name__ == "__main__":
    main()
