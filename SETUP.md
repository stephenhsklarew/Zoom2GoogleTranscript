# Quick Setup Guide

## Installation (5 minutes)

### 1. Install ffmpeg
```bash
brew install ffmpeg
```

### 2. Install Python dependencies
```bash
cd "/Users/stephensklarew/Development/Scripts/Zoom mp4 to transcription"
pip install -r requirements.txt
```

### 3. Verify Google credentials
The script will automatically find your existing Google credentials from the Qwilo project.

✅ If you've already set up Qwilo, you're ready to go!

## First Run

Test with a single video:
```bash
python video_transcriber.py /path/to/your/video.mp4
```

Or batch process a folder:
```bash
python video_transcriber.py /path/to/zoom/videos
```

The first run will download the Whisper model (~140MB for base model).

## Common Commands

```bash
# Default (base model, recursive search)
python video_transcriber.py ~/Zoom/Recordings

# Fast mode for testing
python video_transcriber.py ~/Videos --model tiny

# Best quality
python video_transcriber.py ~/Videos --model medium

# Non-recursive (current folder only)
python video_transcriber.py ~/Videos --no-recursive

# Save to specific Google Drive folder
python video_transcriber.py ~/Videos --folder-id YOUR_FOLDER_ID
```

## Getting Google Drive Folder ID

1. Open folder in Google Drive
2. Copy ID from URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
3. Use with: `--folder-id FOLDER_ID_HERE`

## Troubleshooting

**"ffmpeg not found"**
```bash
brew install ffmpeg
```

**"Could not find credentials file"**
- The script looks in 3 locations automatically
- Make sure you've run Qwilo's cli.py at least once
- Or specify manually: `--credentials /path/to/token.pickle`

**Out of memory**
```bash
python video_transcriber.py ~/Videos --model tiny
```

## Full Documentation

See [README.md](README.md) for complete documentation.
