from __future__ import annotations

import argparse
import base64
import cgi
import io
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime_style.infer import AnimeStylizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the anime style Gradio demo.")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ui", choices=["auto", "gradio", "builtin"], default="auto")
    return parser


def image_data_uri(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_page(input_uri: str = "", output_uri: str = "", error: str = "") -> bytes:
    preview = ""
    if input_uri or output_uri:
        preview = f"""
        <section class="preview">
          <figure>{f'<img src="{input_uri}" alt="Input image">' if input_uri else ''}<figcaption>Input</figcaption></figure>
          <figure>{f'<img src="{output_uri}" alt="Output image">' if output_uri else ''}<figcaption>Output</figcaption></figure>
        </section>
        """
    error_html = f'<p class="error">{error}</p>' if error else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anime Style Generator</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #181a20;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }}
    form {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      background: #ffffff;
    }}
    input[type="file"] {{
      flex: 1;
      min-width: 180px;
    }}
    button {{
      border: 0;
      border-radius: 8px;
      background: #166a5b;
      color: white;
      font-weight: 700;
      padding: 10px 16px;
      cursor: pointer;
    }}
    .preview {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    figure {{
      margin: 0;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #eef1f5;
    }}
    figcaption {{
      padding: 10px 12px;
      font-size: 14px;
      font-weight: 700;
      color: #3b414c;
    }}
    .error {{
      color: #a13030;
      font-weight: 700;
    }}
    @media (max-width: 720px) {{
      header, form {{
        align-items: stretch;
        flex-direction: column;
      }}
      .preview {{
        grid-template-columns: 1fr;
      }}
      button {{
        width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header><h1>Anime Style Generator</h1></header>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Generate</button>
    </form>
    {error_html}
    {preview}
  </main>
</body>
</html>"""
    return html.encode("utf-8")


def run_builtin_server(host: str, port: int, predict: Callable) -> None:
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            self._send(render_page())

        def do_POST(self):
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )
                if "image" not in form:
                    self._send(render_page(error="No image uploaded."))
                    return
                field = form["image"]
                data = field.file.read()
                from PIL import Image

                source = Image.open(io.BytesIO(data)).convert("RGB")
                result = predict(source)
                self._send(render_page(image_data_uri(source), image_data_uri(result)))
            except Exception as exc:
                self._send(render_page(error=str(exc)))

        def _send(self, body: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Running on http://{host}:{port}", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    stylizer = AnimeStylizer(
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
        use_fallback=True,
    )

    def predict(image):
        if image is None:
            return None
        return stylizer.stylize(image)

    gr = None
    if args.ui in {"auto", "gradio"}:
        try:
            import gradio as gr
        except ImportError as exc:
            if args.ui == "gradio":
                raise SystemExit("Gradio is not installed. Run: pip install -r requirements.txt") from exc

    if gr is None:
        run_builtin_server(args.host, args.port, predict)
        return

    with gr.Blocks(title="Anime Style Generator") as demo:
        gr.Markdown("# Anime Style Generator")
        with gr.Row():
            source = gr.Image(type="pil", label="Input photo")
            output = gr.Image(type="pil", label="Anime style output")
        run = gr.Button("Generate", variant="primary")
        run.click(fn=predict, inputs=source, outputs=output)

    try:
        demo.launch(server_name=args.host, server_port=args.port)
    except Exception:
        if args.ui == "gradio":
            raise
        print("Gradio failed to launch; falling back to builtin HTTP UI.", flush=True)
        run_builtin_server(args.host, args.port, predict)


if __name__ == "__main__":
    main()
