from __future__ import annotations

import argparse
import base64
import cgi
import datetime as dt
import io
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anime_style.infer import AnimeStylizer

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "anime_cyclegan" / "latest.pt"
DEFAULT_OUTPUTS = ROOT / "outputs" / "app"


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return ROOT


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Photo2Anime local app.")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="Path to trained checkpoint. Use 'auto' for the local default or 'none' for fallback.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--ui", choices=["auto", "gradio", "builtin"], default="auto")
    parser.add_argument("--outputs", default="outputs/app", help="Directory for generated images.")
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser.")
    return parser


def resolve_checkpoint(value: str | None) -> Path | None:
    root = resource_root()
    default_checkpoint = root / "checkpoints" / "anime_cyclegan" / "latest.pt"
    if value is None or value.lower() == "auto":
        return default_checkpoint if default_checkpoint.exists() else None
    if value.lower() in {"none", "fallback", "off"}:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def resolve_output_dir(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return app_root() / path


def browser_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{browser_host}:{port}/"


def open_browser_later(host: str, port: int) -> None:
    url = browser_url(host, port)
    timer = threading.Timer(1.0, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def safe_stem(name: str) -> str:
    allowed = []
    for char in Path(name).stem:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        elif char.isspace():
            allowed.append("_")
    return "".join(allowed).strip("_") or "image"


def save_generated_image(image, output_dir: str | Path, source_name: str = "image") -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = target_dir / f"{timestamp}_{safe_stem(source_name)}_anime.png"
    image.save(target)
    return target


def image_data_uri(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_page(input_uri: str = "", output_uri: str = "", error: str = "", saved_path: str = "") -> bytes:
    preview = ""
    if input_uri or output_uri:
        preview = f"""
        <section class="preview">
          <figure>{f'<img src="{input_uri}" alt="Input image">' if input_uri else ''}<figcaption>Input</figcaption></figure>
          <figure>{f'<img src="{output_uri}" alt="Output image">' if output_uri else ''}<figcaption>Output</figcaption></figure>
        </section>
        """
    actions = ""
    if output_uri:
        actions = f'<a class="download" href="{output_uri}" download="anime_output.png">Download</a>'
    saved_html = f'<p class="status">Saved to {saved_path}</p>' if saved_path else ""
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
      background: #195d5f;
      color: white;
      font-weight: 700;
      padding: 10px 16px;
      cursor: pointer;
    }}
    .download {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin-top: 16px;
      border-radius: 8px;
      background: #1f2937;
      color: #ffffff;
      font-weight: 700;
      padding: 10px 16px;
      text-decoration: none;
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
    .status {{
      color: #3b414c;
      font-size: 14px;
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
    <header><h1>Photo2Anime</h1></header>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Generate</button>
    </form>
    {error_html}
    {saved_html}
    {actions}
    {preview}
  </main>
</body>
</html>"""
    return html.encode("utf-8")


def run_builtin_server(host: str, port: int, predict: Callable, output_dir: str | Path) -> None:
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
                filename = getattr(field, "filename", "") or "upload"
                saved = save_generated_image(result, output_dir, filename)
                self._send(render_page(image_data_uri(source), image_data_uri(result), saved_path=str(saved)))
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


def launch_gradio_app(host: str, port: int, predict: Callable, output_dir: str | Path, open_app: bool) -> None:
    import gradio as gr
    from PIL import Image

    css = """
    .gradio-container {
      max-width: 1180px !important;
    }
    #app-title h1 {
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
      margin-bottom: 4px;
    }
    .compact-status textarea {
      font-size: 13px !important;
    }
    """

    def predict_single(image):
        if image is None:
            return None, "No image selected."
        result = predict(image.convert("RGB"))
        saved = save_generated_image(result, output_dir, "single")
        return result, f"Saved to {saved}"

    def predict_batch(files):
        if not files:
            return [], "No files selected."

        results = []
        for file in files:
            path = Path(getattr(file, "name", str(file)))
            source = Image.open(path).convert("RGB")
            result = predict(source)
            saved = save_generated_image(result, output_dir, path.name)
            results.append((str(saved), path.name))
        return results, f"Saved {len(results)} image(s) to {Path(output_dir)}"

    with gr.Blocks(title="Photo2Anime", css=css) as demo:
        gr.Markdown("# Photo2Anime", elem_id="app-title")
        with gr.Tabs():
            with gr.Tab("Single"):
                with gr.Row(equal_height=True):
                    source = gr.Image(type="pil", label="Input")
                    output = gr.Image(type="pil", label="Output")
                run = gr.Button("Generate", variant="primary")
                single_status = gr.Textbox(label="Status", interactive=False, elem_classes=["compact-status"])
                run.click(fn=predict_single, inputs=source, outputs=[output, single_status])
            with gr.Tab("Batch"):
                files = gr.Files(label="Images", file_types=["image"], file_count="multiple")
                batch_run = gr.Button("Generate Batch", variant="primary")
                gallery = gr.Gallery(label="Results", columns=3, object_fit="contain", height="auto")
                batch_status = gr.Textbox(label="Status", interactive=False, elem_classes=["compact-status"])
                batch_run.click(fn=predict_batch, inputs=files, outputs=[gallery, batch_status])

    demo.launch(server_name=host, server_port=port, inbrowser=open_app)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = resolve_checkpoint(args.checkpoint)
    output_dir = resolve_output_dir(args.outputs)

    stylizer = AnimeStylizer(
        checkpoint=checkpoint,
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
        if args.open:
            open_browser_later(args.host, args.port)
        run_builtin_server(args.host, args.port, predict, output_dir)
        return

    try:
        launch_gradio_app(args.host, args.port, predict, output_dir, args.open)
    except Exception:
        if args.ui == "gradio":
            raise
        print("Gradio failed to launch; falling back to builtin HTTP UI.", flush=True)
        if args.open:
            open_browser_later(args.host, args.port)
        run_builtin_server(args.host, args.port, predict, output_dir)


if __name__ == "__main__":
    main()
