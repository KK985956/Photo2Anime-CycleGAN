from __future__ import annotations

from web_demo.app import main


if __name__ == "__main__":
    main(
        [
            "--checkpoint",
            "auto",
            "--outputs",
            "outputs/app",
            "--host",
            "127.0.0.1",
            "--port",
            "7860",
            "--ui",
            "builtin",
            "--open",
        ]
    )
