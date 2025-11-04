#!/usr/bin/env python3
"""
Video Transcription Tool - Local Whisper Edition

Transcribes mp4 video files using local Whisper AI and creates Google Doc transcripts.
Uses zero-cost local processing with OpenAI's Whisper model.

Usage:
  python video_transcriber.py /path/to/videos --model base
  python video_transcriber.py /path/to/videos --model medium --recursive
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import whisper
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

console = Console()


class VideoTranscriber:
    """Transcribes videos using local Whisper and creates Google Docs"""

    def __init__(self, model_size="base", credentials_path="token.pickle", hf_token=None):
        """
        Initialize the transcriber

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            credentials_path: Path to Google OAuth credentials
            hf_token: Hugging Face token for speaker diarization (optional)
        """
        self.model_size = model_size
        self.model = None
        self.diarization_pipeline = None
        self.docs_service = None
        self.drive_service = None
        self.calendar_service = None
        self.credentials_path = credentials_path
        self.hf_token = hf_token or os.environ.get('HF_TOKEN')

    def load_model(self):
        """Load the Whisper model and optionally the diarization pipeline"""
        console.print(f"\n[cyan]Loading Whisper model ({self.model_size})...[/cyan]")
        console.print("[yellow]Note: First run will download the model (~100-3000MB depending on size)[/yellow]")

        self.model = whisper.load_model(self.model_size)
        console.print("[green]✓ Whisper model loaded successfully![/green]")

        # Load speaker diarization if HF token is available
        if self.hf_token and PYANNOTE_AVAILABLE:
            console.print("\n[cyan]Loading speaker diarization model...[/cyan]")
            console.print("[yellow]Note: First run will download the diarization model (~300MB)[/yellow]")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_token
                )
                console.print("[green]✓ Speaker diarization enabled![/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load diarization model: {e}[/yellow]")
                console.print("[yellow]Falling back to pause-based speaker detection[/yellow]")
                self.diarization_pipeline = None
        elif not self.hf_token and PYANNOTE_AVAILABLE:
            console.print("[yellow]ℹ Speaker diarization disabled (no HF_TOKEN)[/yellow]")
            console.print("[yellow]  Set HF_TOKEN environment variable or use --hf-token flag for better speaker identification[/yellow]")

        console.print()

    def init_google_services(self):
        """Initialize Google Docs and Drive APIs using existing credentials"""
        # Try multiple locations for credentials
        possible_paths = [
            self.credentials_path,
            os.path.join(os.path.dirname(__file__), 'token.pickle'),
            os.path.expanduser('~/Development/Scripts/Blog idea generator/token.pickle'),
        ]

        creds_path = None
        for path in possible_paths:
            if os.path.exists(path):
                creds_path = path
                console.print(f"[cyan]Using credentials from: {path}[/cyan]")
                break

        if not creds_path:
            console.print(f"[red]Error: Could not find credentials file.[/red]")
            console.print("[yellow]Tried locations:[/yellow]")
            for path in possible_paths:
                console.print(f"  - {path}")
            console.print("\n[yellow]Please run the Qwilo cli.py first to authenticate with Google,[/yellow]")
            console.print("[yellow]or specify credentials path with --credentials flag.[/yellow]")
            sys.exit(1)

        with open(creds_path, 'rb') as token:
            creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(creds_path, 'wb') as token:
                    pickle.dump(creds, token)

        self.docs_service = build('docs', 'v1', credentials=creds)
        self.drive_service = build('drive', 'v3', credentials=creds)
        self.calendar_service = build('calendar', 'v3', credentials=creds)

        console.print("[green]✓ Connected to Google services (Docs, Drive, Calendar)![/green]\n")

    def parse_zoom_folder_datetime(self, video_path: str) -> datetime:
        """
        Parse date/time from Zoom folder name

        Format: "2025-07-02 13.00.28 Meeting Name/video.mp4"

        Args:
            video_path: Full path to video file

        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Get parent folder name
            folder_name = Path(video_path).parent.name

            # Extract date/time part (first 19 characters: "2025-07-02 13.00.28")
            datetime_str = folder_name[:19]

            # Parse: YYYY-MM-DD HH.MM.SS
            return datetime.strptime(datetime_str, '%Y-%m-%d %H.%M.%S')
        except (ValueError, IndexError) as e:
            console.print(f"[yellow]Could not parse date from folder: {folder_name}[/yellow]")
            return None

    def find_calendar_event(self, meeting_datetime: datetime) -> Dict:
        """
        Find calendar event matching the meeting time

        Args:
            meeting_datetime: DateTime of the Zoom recording

        Returns:
            Dict with 'title' and 'attendees' or None
        """
        if not meeting_datetime:
            return None

        try:
            # Search window: 30 minutes before and after
            from datetime import timedelta
            # Convert meeting_datetime (naive local) to UTC by using timestamp
            meeting_utc = datetime.utcfromtimestamp(meeting_datetime.timestamp())
            time_min = (meeting_utc - timedelta(minutes=30)).isoformat() + 'Z'
            time_max = (meeting_utc + timedelta(minutes=30)).isoformat() + 'Z'

            # Query calendar
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=10,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            if not events:
                return None

            # Find closest event
            closest_event = None
            min_diff = float('inf')

            for event in events:
                event_start = event.get('start', {}).get('dateTime')
                if not event_start:
                    continue

                # Parse event time and convert to local time for comparison
                event_time = datetime.fromisoformat(event_start.replace('Z', '+00:00'))
                # Convert to local time by getting the UTC offset from the event
                # and converting meeting_datetime (which is naive local) to UTC for comparison
                event_time_utc = event_time.timestamp()
                meeting_time_utc = meeting_datetime.timestamp()
                diff = abs(event_time_utc - meeting_time_utc)

                if diff < min_diff:
                    min_diff = diff
                    closest_event = event

            if not closest_event:
                return None

            # Extract attendees
            attendees = []
            for attendee in closest_event.get('attendees', []):
                email = attendee.get('email', '')
                name = attendee.get('displayName') or email.split('@')[0]

                # Skip resource calendars
                if 'resource.calendar.google.com' not in email:
                    attendees.append(name)

            # Include organizer
            organizer = closest_event.get('organizer', {})
            organizer_email = organizer.get('email', '')
            organizer_name = organizer.get('displayName') or organizer_email.split('@')[0]

            if organizer_name and organizer_name not in attendees:
                attendees.insert(0, organizer_name)

            return {
                'title': closest_event.get('summary', 'Untitled Meeting'),
                'attendees': attendees,
                'event_time': closest_event.get('start', {}).get('dateTime')
            }

        except Exception as e:
            console.print(f"[yellow]Calendar lookup failed: {e}[/yellow]")
            return None

    def find_videos(self, root_folder: str, recursive: bool = True) -> List[str]:
        """
        Find all mp4 files in the folder

        Args:
            root_folder: Root directory to search
            recursive: Search subdirectories recursively

        Returns:
            List of video file paths
        """
        pattern = "**/*.mp4" if recursive else "*.mp4"
        videos = list(Path(root_folder).glob(pattern))
        return [str(v) for v in videos]

    def transcribe_video(self, video_path: str) -> Dict:
        """
        Transcribe a video file using Whisper

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary with transcription results
        """
        result = self.model.transcribe(
            video_path,
            verbose=False,
            language="en"  # Can be None for auto-detection
        )

        return result

    def diarize_video(self, video_path: str) -> Optional[Dict]:
        """
        Run speaker diarization on a video file

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary mapping time ranges to speaker IDs, or None if diarization unavailable
        """
        if not self.diarization_pipeline:
            return None

        try:
            # Run diarization
            diarization = self.diarization_pipeline(video_path)

            # Convert to dict format: {(start, end): speaker_id}
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments[(turn.start, turn.end)] = speaker

            return speaker_segments

        except Exception as e:
            console.print(f"[yellow]Diarization failed: {e}[/yellow]")
            return None

    def format_transcript_google_meet_style(self, result: Dict, attendees: List[str] = None,
                                           diarization: Optional[Dict] = None) -> str:
        """
        Format transcript in Google Meet style (continuous dialogue)

        Args:
            result: Whisper transcription result
            attendees: List of attendee names from calendar (optional)
            diarization: Speaker diarization data (optional)

        Returns:
            Formatted transcript text in Google Meet style
        """
        lines = []

        # Group segments into speaker turns
        segments = result.get('segments', [])
        if not segments:
            return "No speech detected."

        # Use diarization if available
        if diarization:
            # Map diarization speaker IDs to attendee names
            unique_speakers = sorted(set(diarization.values()))
            speaker_map = {}

            if attendees and len(attendees) >= len(unique_speakers):
                # Map speakers to attendees (SPEAKER_00 -> first attendee, etc.)
                for i, speaker_id in enumerate(unique_speakers):
                    speaker_map[speaker_id] = attendees[i] if i < len(attendees) else f"Speaker {i+1}"
            else:
                # Fallback to numbered speakers
                for i, speaker_id in enumerate(unique_speakers):
                    speaker_map[speaker_id] = f"Speaker {i+1}"

            # Assign each segment to a speaker based on diarization
            for segment in segments:
                text = segment['text'].strip()
                segment_mid = (segment['start'] + segment['end']) / 2

                # Find which diarization segment this belongs to
                speaker_id = None
                for (start, end), spk in diarization.items():
                    if start <= segment_mid <= end:
                        speaker_id = spk
                        break

                if speaker_id:
                    speaker_name = speaker_map.get(speaker_id, "Unknown Speaker")
                else:
                    speaker_name = "Unknown Speaker"

                lines.append(f"{speaker_name}: {text}")

        else:
            # Fallback to pause-based heuristic
            if attendees and len(attendees) >= 2:
                speaker_names = attendees[:2]
            else:
                speaker_names = ["Speaker 1", "Speaker 2"]

            current_speaker_idx = 0
            last_end = 0

            for i, segment in enumerate(segments):
                text = segment['text'].strip()

                # Detect speaker change based on pause (>2 seconds)
                if i > 0:
                    pause_duration = segment['start'] - last_end
                    if pause_duration > 2.0:
                        current_speaker_idx = 1 if current_speaker_idx == 0 else 0

                lines.append(f"{speaker_names[current_speaker_idx]}: {text}")
                last_end = segment['end']

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format (Google Meet style)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def create_google_doc(self, title: str, transcript_result: Dict, video_path: str,
                         folder_id: str = None, calendar_event: Dict = None,
                         diarization: Optional[Dict] = None) -> str:
        """
        Create a Google Doc with the transcript

        Args:
            title: Document title (fallback if no calendar event)
            transcript_result: Whisper transcription result
            video_path: Original video file path
            folder_id: Optional Google Drive folder ID
            calendar_event: Optional calendar event info with 'title' and 'attendees'
            diarization: Optional speaker diarization data

        Returns:
            Document ID
        """
        # Create the document
        doc = self.docs_service.documents().create(body={
            'title': title
        }).execute()

        doc_id = doc['documentId']

        # Move to folder if specified
        if folder_id:
            try:
                self.drive_service.files().update(
                    fileId=doc_id,
                    addParents=folder_id,
                    fields='id, parents'
                ).execute()
            except HttpError as e:
                console.print(f"[yellow]Warning: Could not move to folder: {e}[/yellow]")

        # Format the content
        video_name = Path(video_path).name
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)

        # Get duration safely
        segments = transcript_result.get('segments', [])
        if segments:
            duration_min = segments[-1].get('end', 0) / 60
        else:
            duration_min = 0

        # Build document content - Google Meet style
        # Format: Date, Title - Transcript, Starting timestamp, blank line, then dialogue
        current_date = datetime.now().strftime('%b %d, %Y')  # e.g., "Oct 31, 2025"

        # Use calendar event title if available, otherwise use video filename
        if calendar_event and calendar_event.get('title'):
            meeting_title = calendar_event['title']
        else:
            meeting_title = Path(video_path).stem

        doc_title = f"{meeting_title} - Transcript"

        # Get attendee names from calendar event (for speaker identification)
        attendees = calendar_event.get('attendees', []) if calendar_event else []

        content = f"{current_date}\n"
        content += f"{doc_title}\n"

        # Add attendee list if available
        if attendees:
            attendee_list = ", ".join(attendees)
            content += f"Attendees: {attendee_list}\n"

        content += "00:00:00\n"
        content += " \n"  # Blank line (with space to preserve it)
        content += self.format_transcript_google_meet_style(transcript_result, attendees, diarization)

        # Insert content into document
        requests = [{
            'insertText': {
                'location': {'index': 1},
                'text': content
            }
        }]

        self.docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': requests}
        ).execute()

        return doc_id

    def process_videos(self, root_folder: str, recursive: bool = True,
                      folder_id: str = None) -> List[Dict]:
        """
        Process all videos in the folder

        Args:
            root_folder: Root directory containing videos
            recursive: Search subdirectories
            folder_id: Optional Google Drive folder ID for output docs

        Returns:
            List of processing results
        """
        # Find videos
        console.print(f"[cyan]Scanning for videos in: {root_folder}[/cyan]")
        videos = self.find_videos(root_folder, recursive)

        if not videos:
            console.print("[yellow]No mp4 files found![/yellow]")
            return []

        console.print(f"[green]Found {len(videos)} video(s)[/green]\n")

        # Display video list
        table = Table(title="Videos to Process")
        table.add_column("No.", style="cyan", width=4)
        table.add_column("Filename", style="magenta")
        table.add_column("Size", style="yellow", width=10)

        for idx, video_path in enumerate(videos, 1):
            size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            table.add_row(str(idx), Path(video_path).name, f"{size_mb:.1f} MB")

        console.print(table)
        console.print()

        # Process each video
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            overall_task = progress.add_task(
                f"[cyan]Processing {len(videos)} videos...",
                total=len(videos)
            )

            for idx, video_path in enumerate(videos, 1):
                video_name = Path(video_path).stem

                try:
                    # Look up calendar event for this video
                    calendar_event = None
                    meeting_datetime = self.parse_zoom_folder_datetime(video_path)
                    if meeting_datetime:
                        calendar_event = self.find_calendar_event(meeting_datetime)
                        if calendar_event:
                            console.print(f"[green]✓ Found calendar event: {calendar_event['title']}[/green]")
                        else:
                            console.print(f"[yellow]No calendar event found for {meeting_datetime.strftime('%Y-%m-%d %H:%M')}[/yellow]")

                    # Update progress
                    progress.update(
                        overall_task,
                        description=f"[cyan]Transcribing ({idx}/{len(videos)}): {Path(video_path).name}"
                    )

                    # Transcribe
                    transcript_result = self.transcribe_video(video_path)

                    # Run speaker diarization if enabled
                    diarization = None
                    if self.diarization_pipeline:
                        progress.update(
                            overall_task,
                            description=f"[cyan]Diarizing ({idx}/{len(videos)}): {Path(video_path).name}"
                        )
                        diarization = self.diarize_video(video_path)
                        if diarization:
                            num_speakers = len(set(diarization.values()))
                            console.print(f"[green]✓ Identified {num_speakers} speaker(s)[/green]")

                    # Create Google Doc
                    progress.update(
                        overall_task,
                        description=f"[cyan]Creating Doc ({idx}/{len(videos)}): {video_name}"
                    )

                    # Document title (just the video name, we add "- Transcript" inside)
                    title = video_name
                    doc_id = self.create_google_doc(
                        title,
                        transcript_result,
                        video_path,
                        folder_id,
                        calendar_event,
                        diarization
                    )

                    doc_url = f"https://docs.google.com/document/d/{doc_id}"

                    # Get duration safely
                    segments = transcript_result.get('segments', [])
                    duration = segments[-1].get('end', 0) if segments else 0

                    results.append({
                        'video_path': video_path,
                        'video_name': video_name,
                        'doc_id': doc_id,
                        'doc_url': doc_url,
                        'status': 'success',
                        'duration': duration
                    })

                    console.print(f"[green]✓ {video_name} → {doc_url}[/green]")

                except Exception as e:
                    console.print(f"[red]✗ Error processing {video_name}: {e}[/red]")
                    results.append({
                        'video_path': video_path,
                        'video_name': video_name,
                        'status': 'failed',
                        'error': str(e)
                    })

                progress.update(overall_task, advance=1)

        return results

    def print_summary(self, results: List[Dict]):
        """Print processing summary"""
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        total_duration = sum(r.get('duration', 0) for r in successful) / 60

        summary = f"\n[bold cyan]Processing Complete![/bold cyan]\n\n"
        summary += f"Total Videos: {len(results)}\n"
        summary += f"[green]Successful: {len(successful)}[/green]\n"
        summary += f"[red]Failed: {len(failed)}[/red]\n"
        summary += f"Total Duration: {total_duration:.1f} minutes\n"

        console.print(Panel(summary, title="Summary", border_style="cyan"))

        if successful:
            console.print("\n[bold green]Successfully Created Documents:[/bold green]")
            for r in successful:
                console.print(f"  • {r['video_name']}: {r['doc_url']}")

        if failed:
            console.print("\n[bold red]Failed:[/bold red]")
            for r in failed:
                console.print(f"  • {r['video_name']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe videos to Google Docs using local Whisper AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Sizes:
  tiny    - Fastest, lowest accuracy (~1GB RAM, ~32x realtime)
  base    - Good balance (~1GB RAM, ~16x realtime) [DEFAULT]
  small   - Better accuracy (~2GB RAM, ~6x realtime)
  medium  - High accuracy (~5GB RAM, ~2x realtime)
  large   - Best accuracy (~10GB RAM, ~1x realtime)

Speaker Diarization (optional):
  For accurate speaker identification, get a free Hugging Face token:
  1. Create account at https://huggingface.co
  2. Accept agreement at https://huggingface.co/pyannote/speaker-diarization-3.1
  3. Get token from https://huggingface.co/settings/tokens
  4. Use --hf-token flag or set HF_TOKEN environment variable

Examples:
  python video_transcriber.py /path/to/videos
  python video_transcriber.py /path/to/videos --model medium
  python video_transcriber.py /path/to/videos --hf-token hf_xxxxx
  export HF_TOKEN=hf_xxxxx && python video_transcriber.py /path/to/videos
  python video_transcriber.py /path/to/videos --no-recursive
  python video_transcriber.py /path/to/videos --folder-id abc123xyz
        """
    )

    parser.add_argument(
        'video_folder',
        help='Path to folder containing mp4 videos'
    )
    parser.add_argument(
        '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model size (default: base)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )
    parser.add_argument(
        '--folder-id',
        help='Google Drive folder ID to save documents in'
    )
    parser.add_argument(
        '--credentials',
        default='token.pickle',
        help='Path to Google credentials file (default: token.pickle)'
    )
    parser.add_argument(
        '--hf-token',
        help='Hugging Face token for speaker diarization (can also use HF_TOKEN env variable)'
    )

    args = parser.parse_args()

    # Validate video folder
    if not os.path.isdir(args.video_folder):
        console.print(f"[red]Error: {args.video_folder} is not a valid directory[/red]")
        sys.exit(1)

    # Display banner
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║           VIDEO TRANSCRIPTION TOOL                       ║
║           Powered by Local Whisper AI                    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")

    # Initialize transcriber
    transcriber = VideoTranscriber(
        model_size=args.model,
        credentials_path=args.credentials,
        hf_token=args.hf_token
    )

    # Load model
    transcriber.load_model()

    # Initialize Google services
    transcriber.init_google_services()

    # Process videos
    results = transcriber.process_videos(
        args.video_folder,
        recursive=not args.no_recursive,
        folder_id=args.folder_id
    )

    # Print summary
    transcriber.print_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
