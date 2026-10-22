"""A stand-in for GitHub's releases, for the update test: one release with one installer and its manifest.

  python packaging/fake_releases.py --tag fx-v1.0.7.1 --file dist-next/BokinFX-Setup-1.0.7.1.exe [--port 8765]

serves http://127.0.0.1:<port>/releases like api.github.com/repos/<owner>/<repo>/releases, with the file's size
and SHA-256 digest as GitHub lists them, and http://127.0.0.1:<port>/site like github.com/<owner>/<repo>: the newest
release's manifest at /site/releases/latest/download/bokin-fx-latest.json, its installer at
/site/releases/download/<tag>/<name>. /limited answers as the API does once an Internet connection has used up its
requests (403, X-RateLimit-Remaining: 0). Download addresses answer with a redirect to another host name
(localhost), as GitHub hands downloads to its file servers. serve() does the same from Python, in a thread.
"""
import argparse
import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve(tag, path, port):
    name, size = os.path.basename(path), os.path.getsize(path)
    with open(path, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    listing = json.dumps([{"tag_name": tag, "draft": False, "prerelease": False, "body": "",
                           "assets": [{"name": name, "size": size, "digest": f"sha256:{sha}",
                                       "browser_download_url": f"http://127.0.0.1:{port}/get/{name}"}]}]).encode()
    manifest = json.dumps({"version": tag[len("fx-v"):], "name": name, "size": size, "sha256": sha}).encode()

    class Answer(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print("fake GitHub:", fmt % args, flush=True)

        def reply(self, code, body=b"", headers=()):
            self.send_response(code)
            for k, v in headers:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/releases":
                self.reply(200, listing, [("Content-Type", "application/json")])
            elif route == "/limited":
                self.reply(403, b'{"message": "API rate limit exceeded"}', [("X-RateLimit-Remaining", "0")])
            elif route == "/site/releases/latest/download/bokin-fx-latest.json":
                self.reply(302, headers=[("Location", f"http://localhost:{port}/blob/bokin-fx-latest.json")])
            elif route == "/blob/bokin-fx-latest.json":
                self.reply(200, manifest, [("Content-Type", "application/octet-stream")])
            elif route in (f"/get/{name}", f"/site/releases/download/{tag}/{name}"):
                self.reply(302, headers=[("Location", f"http://localhost:{port}/blob/{name}")])
            elif route == f"/blob/{name}":
                with open(path, "rb") as f:
                    self.reply(200, f.read(), [("Content-Type", "application/octet-stream")])
            else:
                self.reply(404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    serve(args.tag, args.file, args.port)
    threading.Event().wait()
