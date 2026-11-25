#!/usr/bin/env python3
"""notreal-server: the small AI server your notreal phone talks to.

Runs on a Mac, Windows or Linux computer you own. Python 3.8+, no packages to install.
It sits in front of Ollama (https://ollama.com), which does the thinking, and gives the phone:

    GET  /health               the phone's "is my computer there?" check
    GET  /v1/models            model list
    POST /v1/chat/completions  summaries, questions, the digest (forwarded to Ollama)
    POST /fetch                reads a web page so the assistant can summarise a pasted link

Everything else the full notreal setup offers (voice typing on the computer, photo enhance, pictures,
web search, the video hub) answers 501 here, and the phone falls back to doing it itself.

    python3 notreal-server.py --install     # first time: picks a model, sets it to start at login
    python3 notreal-server.py               # run it now, in this window
    python3 notreal-server.py --uninstall

Options (or environment variables in capitals, e.g. NOTREAL_MODEL):
    --model NAME       Ollama model to use (default: the first one installed, else gemma3:4b is downloaded)
    --listen ADDR      address to listen on (default: this computer's Tailscale address)
    --port N           default 8097 (the phone expects this)
    --ollama URL       default http://127.0.0.1:11434
"""
import argparse
import html.parser
import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODEL = "gemma3:4b"      # small, no "thinking" phase, good enough for summaries; ~3 GB
NAME = "notreal-server"
cfg = {"model": "", "ollama": "http://127.0.0.1:11434"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- Ollama

def ollama(path, body=None, timeout=300, method=None):
    req = urllib.request.Request(cfg["ollama"] + path, data=body, method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def installed_models():
    try:
        return [m["name"] for m in json.loads(ollama("/api/tags", timeout=5)).get("models", [])]
    except Exception:
        return None          # None = Ollama itself is not answering


def pick_model():
    """The model to use: the one asked for, else the first installed one, else the default (downloaded by --install)."""
    have = installed_models() or []
    want = cfg["model"]
    if want:
        return want
    return have[0] if have else DEFAULT_MODEL


def ensure_model():
    have = installed_models()
    if have is None:
        sys.exit("Ollama is not running. Install it from https://ollama.com, open it once, then run this again.")
    m = pick_model()
    if m not in have and (m + ":latest") not in have:
        exe = shutil.which("ollama")
        if not exe:
            sys.exit("Ollama is running but the 'ollama' command is not on the PATH, so the model cannot be downloaded. Open a new terminal and retry.")
        log(f"downloading {m} (a few GB, once) ...")
        if subprocess.call([exe, "pull", m]) != 0:
            sys.exit(f"could not download {m}")
    return m


# ---------------------------------------------------------------- web page reader

class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"}

    def __init__(self):
        super().__init__()
        self.out, self.depth, self.title, self.in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag == "title":
            self.in_title = True
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "div", "tr", "section", "article"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        elif tag == "title":
            self.in_title = False

    def handle_data(self, d):
        if self.in_title:
            self.title += d
        elif not self.depth and d.strip():
            self.out.append(d)


def fetch_page(url, limit=7000):
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("not an http(s) address")
    for info in socket.getaddrinfo(u.hostname, None):           # never let the phone reach into your own network through us
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("private address")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; notreal-server)", "Accept-Language": "en,sv;q=0.8"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read(2_000_000)
        enc = r.headers.get_content_charset() or "utf-8"
    p = _Text()
    p.feed(raw.decode(enc, "replace"))
    text = re.sub(r"[ \t]+", " ", "".join(p.out))
    return p.title.strip(), re.sub(r"\n\s*\n+", "\n", text).strip()[:limit]


# ---------------------------------------------------------------- the server

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            have = installed_models()
            return self._send(200, {"ok": True, "loaded": have is not None, "model": cfg["model"], "server": NAME})
        if self.path in ("/v1/models", "/models"):
            return self._send(200, {"object": "list", "data": [{"id": cfg["model"], "object": "model", "owned_by": NAME}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        if self.path == "/fetch":
            try:
                url = str(json.loads(body or b"{}").get("url", "")).strip()
                title, text = fetch_page(url)
                return self._send(200, {"url": url, "title": title, "text": text})
            except Exception as e:
                return self._send(503, {"error": str(e)})
        if self.path.startswith("/v1/"):
            try:
                j = json.loads(body or b"{}")
                j["model"] = cfg["model"]                 # the phone never names a model; this server decides
                j.pop("chat_template_kwargs", None)      # llama.cpp-only option the phone sends
                t0 = time.time()
                out = ollama(self.path, json.dumps(j).encode())
                log(f"{self.path} {len(body)}B -> {len(out)}B in {time.time() - t0:.1f}s")
                return self._send(200, out)
            except Exception as e:
                log("error:", repr(e))
                return self._send(503, {"error": str(e)})
        self._send(501, {"error": "not enabled on this server"})


def tailscale_ip():
    for exe in (shutil.which("tailscale"), "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
                r"C:\Program Files\Tailscale\tailscale.exe"):
        if exe and os.path.exists(exe):
            try:
                out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
                if out:
                    return out[0]
            except Exception:
                pass
    return None


# ---------------------------------------------------------------- start at login

def install(args):
    me = os.path.abspath(__file__)
    cmd = [sys.executable, me] + (["--model", args.model] if args.model else []) + (["--listen", args.listen] if args.listen else []) + ["--port", str(args.port)]
    sysname = platform.system()
    if sysname == "Darwin":
        p = os.path.expanduser("~/Library/LaunchAgents/os.notreal.server.plist")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        items = "".join(f"<string>{c}</string>" for c in cmd)
        open(p, "w").write(f'<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict><key>Label</key><string>os.notreal.server</string>'
                           f'<key>ProgramArguments</key><array>{items}</array><key>RunAtLoad</key><true/><key>KeepAlive</key><true/>'
                           f'<key>StandardErrorPath</key><string>/tmp/notreal-server.log</string><key>StandardOutPath</key><string>/tmp/notreal-server.log</string></dict></plist>')
        subprocess.call(["launchctl", "unload", p], stderr=subprocess.DEVNULL)
        subprocess.call(["launchctl", "load", p])
    elif sysname == "Windows":
        run = " ".join(f'"{c}"' for c in cmd)
        subprocess.call(["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", "notreal-server", "/TR", run, "/RL", "LIMITED"])
        subprocess.Popen(cmd, creationflags=0x00000008)     # DETACHED_PROCESS: start now too
    else:
        d = os.path.expanduser("~/.config/systemd/user")
        os.makedirs(d, exist_ok=True)
        exec_ = " ".join(f'"{c}"' for c in cmd)
        open(os.path.join(d, "notreal-server.service"), "w").write(
            f"[Unit]\nDescription=notreal-server\nAfter=network-online.target\n\n[Service]\nExecStart={exec_}\nRestart=always\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n")
        subprocess.call(["systemctl", "--user", "daemon-reload"])
        subprocess.call(["systemctl", "--user", "enable", "--now", "notreal-server.service"])
    log("installed: notreal-server now starts when you log in.")


def uninstall():
    sysname = platform.system()
    if sysname == "Darwin":
        p = os.path.expanduser("~/Library/LaunchAgents/os.notreal.server.plist")
        subprocess.call(["launchctl", "unload", p])
        if os.path.exists(p):
            os.remove(p)
    elif sysname == "Windows":
        subprocess.call(["schtasks", "/Delete", "/F", "/TN", "notreal-server"])
    else:
        subprocess.call(["systemctl", "--user", "disable", "--now", "notreal-server.service"])
        p = os.path.expanduser("~/.config/systemd/user/notreal-server.service")
        if os.path.exists(p):
            os.remove(p)
    log("removed.")


def main():
    ap = argparse.ArgumentParser(description="the small AI server for your notreal phone")
    ap.add_argument("--install", action="store_true", help="pick/download a model and start at every login")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--model", default=os.environ.get("NOTREAL_MODEL", ""))
    ap.add_argument("--listen", default=os.environ.get("NOTREAL_LISTEN", ""))
    ap.add_argument("--port", type=int, default=int(os.environ.get("NOTREAL_PORT", "8097")))
    ap.add_argument("--ollama", default=os.environ.get("NOTREAL_OLLAMA", cfg["ollama"]))
    a = ap.parse_args()
    cfg["ollama"] = a.ollama.rstrip("/")
    cfg["model"] = a.model
    if a.uninstall:
        return uninstall()
    cfg["model"] = ensure_model()
    if a.install:
        a.model = cfg["model"]
        install(a)
        host = a.listen or tailscale_ip() or "(no Tailscale found)"
        log(f"phone address to type into notreal: {host}")
        return
    host = a.listen or tailscale_ip()
    if not host:
        host = "127.0.0.1"
        log("WARNING: Tailscale not found, so only this computer can reach the server. Install Tailscale (tailscale.com/download), sign in, and restart.")
    log(f"{NAME} using model {cfg['model']} on {host}:{a.port}")
    ThreadingHTTPServer((host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
