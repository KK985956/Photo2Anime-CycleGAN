# Release Notes

## Photo2Anime Trained Checkpoint

The release includes the trained model checkpoint:

```text
latest.pt
```

Training images are not included. To use the checkpoint, download `latest.pt` and place it at:

```text
checkpoints/anime_cyclegan/latest.pt
```

Then run the local app:

```powershell
.\start_app.bat
```

or run inference directly:

```powershell
.\.venv\Scripts\python.exe scripts\infer.py data\testA --output outputs\anime --checkpoint checkpoints\anime_cyclegan\latest.pt
```
