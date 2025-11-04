# Zoom2GoogleTranscript

Automatically transcribe Zoom mp4 videos to Google Docs using local Whisper AI with intelligent calendar integration and speaker identification. **Zero ongoing costs** - all processing happens on your machine.

## ✨ Key Features

- 🎥 **Batch Processing** - Transcribe multiple videos automatically
- 📅 **Calendar Integration** - Automatically fetches meeting titles and attendees from Google Calendar
- 👥 **Speaker Identification** - Uses calendar attendees for accurate speaker names
- 💰 **Zero Cost** - Uses local Whisper AI (no API charges)
- 📝 **Google Meet Format** - Creates properly formatted transcripts matching Google Meet style
- 🔄 **Progress Tracking** - Rich terminal UI with progress bars
- 🎯 **Model Selection** - Choose speed vs accuracy trade-off
- 🤖 **AI Speaker Diarization** - Optional advanced speaker detection with pyannote.audio

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **ffmpeg** - Required for audio processing
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg
   ```

3. **Google Cloud Project** - For API access
   - Create project at https://console.cloud.google.com
   - Enable Google Docs API, Drive API, and Calendar API
   - Download OAuth 2.0 credentials as `credentials.json`

### Installation

```bash
# Clone the repository
git clone https://github.com/stephenhsklarew/Zoom2GoogleTranscript.git
cd Zoom2GoogleTranscript

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google
python authenticate.py
```

### Basic Usage

```bash
# Transcribe all videos in a folder
python video_transcriber.py /path/to/zoom/recordings

# Use a specific model
python video_transcriber.py /path/to/zoom/recordings --model medium

# Specify credentials
python video_transcriber.py /path/to/zoom/recordings --credentials token_video.pickle
```

## 📋 Output Format

The tool creates Google Docs transcripts in Google Meet format:

```
Dec 9, 2024
Steve/Karan/Stephen - Transcript
Attendees: karan.apatel, stephen.sklarew, steve.burden
00:00:00

karan.apatel: Hey everyone, thanks for joining...
stephen.sklarew: Great to be here. Let's discuss...
steve.burden: I'll start with the quarterly results...
```

### How It Works

1. **Extracts date/time** from Zoom folder names (format: `YYYY-MM-DD HH.MM.SS Meeting Name`)
2. **Queries Google Calendar** for matching events (±30 minute window)
3. **Extracts meeting details** - title and attendee list
4. **Transcribes audio** using Whisper AI
5. **Maps speakers** to calendar attendees
6. **Creates formatted Google Doc** with proper attribution

## 🎤 Speaker Identification

### Method 1: Calendar-Based (Default)

Uses pause detection (>2 seconds) combined with calendar attendee names:

- ✅ Zero setup required
- ✅ Works immediately with calendar integration
- ✅ Good for 2-3 person conversations
- ⚠️ Less accurate for complex multi-speaker scenarios

### Method 2: AI-Powered Diarization (Optional)

For advanced speaker detection with pyannote.audio:

1. **Get Hugging Face Token:**
   - Create account at https://huggingface.co
   - Accept agreement at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Get token from https://huggingface.co/settings/tokens

2. **Use with token:**
   ```bash
   # Via environment variable (recommended)
   export HF_TOKEN=hf_your_token_here
   python video_transcriber.py /path/to/videos

   # Or via command line
   python video_transcriber.py /path/to/videos --hf-token hf_your_token_here
   ```

**Benefits of AI Diarization:**
- 🎯 More accurate speaker detection
- 👥 Better for 3+ person meetings
- 🔊 Analyzes voice characteristics, not just pauses
- ✅ Still free (runs locally)

## 🎛️ Command Line Options

```bash
python video_transcriber.py <video_folder> [OPTIONS]

Required:
  video_folder              Path to folder containing Zoom recordings

Optional:
  --model MODEL            Whisper model: tiny, base, small, medium, large
                           (default: base)

  --no-recursive           Don't search subdirectories

  --folder-id ID           Google Drive folder ID to save documents

  --credentials PATH       Path to Google credentials file
                           (default: token_video.pickle)

  --hf-token TOKEN         Hugging Face token for speaker diarization
                           (can also use HF_TOKEN environment variable)
```

## 📊 Model Comparison

| Model | Speed | Accuracy | RAM | Download Size |
|-------|-------|----------|-----|---------------|
| **tiny** | ~32x realtime | Lowest | 1GB | ~75MB |
| **base** | ~16x realtime | Good ✅ | 1GB | ~140MB |
| **small** | ~6x realtime | Better | 2GB | ~460MB |
| **medium** | ~2x realtime | High | 5GB | ~1.5GB |
| **large** | ~1x realtime | Best | 10GB | ~3GB |

**Recommendation:** Start with `base` model for speed/quality balance.

### Real-World Performance

MacBook Pro M1 (CPU only):
- **30 min video** with `base` model: ~2 minutes
- **30 min video** with `medium` model: ~4 minutes

## 🔧 Setup Details

### 1. Google Cloud Setup

1. Go to https://console.cloud.google.com
2. Create a new project
3. Enable these APIs:
   - Google Docs API
   - Google Drive API
   - Google Calendar API (v3)
4. Create OAuth 2.0 credentials:
   - Application type: "Desktop app"
   - Download as `credentials.json`
   - Place in project directory

### 2. Authentication

Run the authentication script once:

```bash
python authenticate.py
```

This will:
- Open your browser for Google OAuth
- Request permissions for Docs, Drive, and Calendar
- Save credentials to `token_video.pickle`

The token is reused for all future transcriptions.

### 3. Zoom Recording Structure

For calendar integration to work, organize recordings in Zoom's default format:

```
Zoom/
├── 2024-12-09 10.31.25 Steve_Karan_Stephen/
│   └── video1487928882.mp4
├── 2024-12-02 15.00.18 Diane_Stephen Weekly 1_1/
│   └── video1683623283.mp4
└── ...
```

The folder name format `YYYY-MM-DD HH.MM.SS Meeting Name` is used to match calendar events.

## 💡 Usage Examples

### Example 1: Weekly Meeting Transcripts

```bash
# Transcribe all recordings from last week
python video_transcriber.py ~/Documents/Zoom --model base

# Review transcripts in Google Docs
# Speaker names automatically pulled from calendar
```

### Example 2: Client Call Archive

```bash
# Process all client recordings with better accuracy
python video_transcriber.py ~/Videos/ClientCalls \
  --model medium \
  --folder-id abc123xyz

# All transcripts organized in specific Drive folder
```

### Example 3: Conference Recording

```bash
# Use AI speaker diarization for multi-speaker panel
export HF_TOKEN=hf_your_token
python video_transcriber.py ~/Conferences/2024 \
  --model medium \
  --recursive
```

## 🔒 Security & Privacy

- ✅ **All AI processing is local** - Videos never sent to external servers
- ✅ **No OpenAI API calls** - Zero data sent to cloud
- ✅ **Google OAuth** - Secure authentication flow
- ✅ **Minimal permissions** - Only Docs/Drive/Calendar access
- ✅ **Token stored locally** - credentials.json and token.pickle stay on your machine

## 🐛 Troubleshooting

### "ffmpeg not found"
```bash
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu/Debian
```

### "Calendar API has not been used"
Enable Calendar API in Google Cloud Console:
https://console.cloud.google.com/apis/library/calendar-json.googleapis.com

### "No calendar event found"
Check that:
- Video folder follows Zoom naming: `YYYY-MM-DD HH.MM.SS Meeting Name`
- Calendar event exists within ±30 minutes of recording time
- Calendar API is enabled and authenticated

### Slow processing
- Use smaller model (`--model base` or `--model tiny`)
- Enable GPU if available (automatic)
- Process overnight for large batches

### Incorrect speaker names
- Verify calendar event has attendees listed
- Try AI diarization with `--hf-token` for better accuracy
- Check that Zoom folder timestamp matches meeting time

## 📝 Cost Comparison

| Solution | Cost | Processing Speed | Accuracy |
|----------|------|------------------|----------|
| **This Tool (Zoom2GoogleTranscript)** | $0 | Local (2-10x realtime) | High |
| Whisper API | $0.006/min | Very Fast | High |
| Google Speech-to-Text | $0.016/min | Very Fast | Medium |
| Rev.ai | $1.50/min | Fast | Very High |

**100 hours of video:**
- Zoom2GoogleTranscript: **$0**
- Whisper API: $36
- Google Speech-to-Text: $96
- Rev.ai: $9,000

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Support for additional video formats (mov, avi, webm)
- Parallel processing for faster batch jobs
- Custom speaker name mapping
- Integration with other calendar systems
- Improved speaker diarization algorithms

## 📄 License

MIT License - Free for personal and commercial use.

## 👤 Author

Stephen Sklarew ([@stephenhsklarew](https://github.com/stephenhsklarew))

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Google APIs](https://developers.google.com/) - Docs, Drive, Calendar integration

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Email: stephen@synaptiq.ai
