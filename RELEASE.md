# Release Notes

## Photo2Anime Windows App

The Windows app is built from `Photo2Anime.spec` and launches the built-in web interface at:

```text
http://127.0.0.1:7860/
```

Users should download the full packaged `Photo2Anime` folder, then run:

```text
Photo2Anime.exe
```

or:

```text
启动 Photo2Anime.bat
```

Do not distribute only the `.exe`; the `_internal` folder contains required Python, PyTorch, and checkpoint files.

### Compatibility

- Windows x64.
- No local Python installation required.
- NVIDIA GPU is optional. CPU fallback works but is slower.
- The included model checkpoint controls output quality and style.

### Large Files

The packaged app is not tracked in Git because it contains multi-GB runtime files and model weights. Upload packaged builds to GitHub Releases or another file hosting service.
