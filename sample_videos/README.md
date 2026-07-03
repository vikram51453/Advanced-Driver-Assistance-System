# Sample Videos

Video files are not included in this repository.

Place your own dashcam or road footage here before running the pipeline.

**Supported formats:** `.mp4`, `.avi`

The pipeline automatically scans this folder at startup and loads all video files it finds — no configuration or hardcoded paths needed. Just drop your video in here and run:

```bash
python run_adas.py
```

If this folder is empty, the pipeline will prompt you to add a video before it can start.

---

**Good sources for test footage:**
- Your own dashcam recordings
- Publicly available driving datasets (BDD100K, KITTI, nuScenes)
- Any dashcam footage from YouTube downloaded as `.mp4`