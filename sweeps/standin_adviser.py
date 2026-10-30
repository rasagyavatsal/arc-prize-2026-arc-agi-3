"""Stand-in adviser server: the control any model has to beat.

Speaks the OpenAI chat shape the Explorer's HttpAdviser uses and answers
every prompt the same trivial way: click the three smallest listed objects
first, no move preference.  Reordering clicks alone moves dev by about a
level, so a model's dev gain only counts if it beats this.

Usage:  python sweeps/standin_adviser.py [PORT]   (then ADVISER_URL=http://127.0.0.1:PORT/v1)
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = json.loads(body["messages"][-1]["content"])
        answer = {"click_first": [o["id"] for o in prompt.get("objects", [])[:3]], "move_first": []}
        out = json.dumps({"choices": [{"message": {"content": json.dumps(answer)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8100
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
