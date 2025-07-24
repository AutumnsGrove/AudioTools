#!/usr/bin/env python3

import os
import subprocess
import tempfile
import gradio as gr
from pathlib import Path
from datetime import datetime

# Configuration
TEMP_DIR = "/tmp/video_transcribe"
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"

def ensure_directories():
    """Create temporary directory if it doesn't exist."""
    os.makedirs(TEMP_DIR, exist_ok=True)

def get_file_metadata(file_path):
    """Get file creation and modification times."""
    try:
        stat = os.stat(file_path)
        # Use modification time as it's more reliable across systems
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        return mod_time.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "Unknown"

def convert_to_mp3(video_path):
    """Convert video file to MP3 format."""
    video_name = Path(video_path).stem
    mp3_path = os.path.join(TEMP_DIR, f"{video_name}.mp3")
    
    cmd = [
        FFMPEG_PATH,
        "-y",                 # Overwrite output file if exists
        "-i", video_path,     # Input file
        "-q:a", "0",         # Best quality
        "-map", "a",         # Extract audio only
        mp3_path             # Output file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return mp3_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error converting video to MP3: {e.stderr}")

def transcribe_audio(mp3_path):
    """Transcribe MP3 file using parakeet-mlx."""
    try:
        cmd = [
            "parakeet-mlx",
            "--output-format", "txt",
            "--output-dir", TEMP_DIR,
            mp3_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Get the output file path
        txt_path = os.path.join(TEMP_DIR, f"{Path(mp3_path).stem}.txt")
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                transcript = f.read()
            os.remove(txt_path)  # Clean up the temporary txt file
            return transcript
        else:
            raise Exception("Transcription file was not created")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during transcription: {e.stderr}")

def process_single_video(video_file, progress_callback=None):
    """Process a single video file and return formatted transcript."""
    video_path = video_file.name
    video_name = Path(video_path).stem
    file_timestamp = get_file_metadata(video_path)
    
    try:
        # Convert video to MP3
        mp3_path = convert_to_mp3(video_path)
        
        try:
            # Transcribe audio
            transcript = transcribe_audio(mp3_path)
            
            # Format the transcript with metadata
            formatted_transcript = f"""# {video_name}
**File Date:** {file_timestamp}

{transcript}

---

"""
            return formatted_transcript, None
            
        finally:
            # Clean up MP3 file
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
                
    except Exception as e:
        error_msg = f"""# {video_name}
**File Date:** {file_timestamp}
**Error:** {str(e)}

---

"""
        return error_msg, str(e)

def process_videos(video_files, progress=gr.Progress()):
    """Main processing function for multiple video files."""
    if not video_files or len(video_files) == 0:
        return "Please upload one or more video files."
    
    ensure_directories()
    
    all_transcripts = []
    total_files = len(video_files)
    
    # Add header with summary
    summary = f"""# Video Transcription Results
**Total Files:** {total_files}
**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    all_transcripts.append(summary)
    
    for i, video_file in enumerate(video_files):
        progress_percent = (i / total_files)
        progress(progress_percent, desc=f"Processing {Path(video_file.name).name}")
        
        transcript, error = process_single_video(video_file)
        all_transcripts.append(transcript)
        
        # Clean up the uploaded file copy immediately after processing
        try:
            if os.path.exists(video_file.name):
                os.remove(video_file.name)
        except:
            pass  # Don't fail if we can't clean up
    
    progress(1.0, desc="Complete!")
    return "".join(all_transcripts)

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Define the interface
    interface = gr.Interface(
        fn=process_videos,
        inputs=gr.File(
            file_types=["video"],
            label="Upload Video Files",
            file_count="multiple",
            height=200
        ),
        outputs=gr.Textbox(
            label="Transcripts",
            lines=30,
            max_lines=100,
            placeholder="Upload video files to see transcripts here...\n\nTip: You can drag and drop multiple files at once for batch processing.",
            show_copy_button=True,
            autoscroll=False
        ),
        title="🎥 Video Transcription Tool",
        description="""
        Upload one or more video files to automatically generate transcripts using parakeet-mlx.
        Files are processed sequentially and results include original file timestamps.
        
        **Note:** Uploaded files are temporarily copied for processing and cleaned up afterwards.
        """,
        article="Supported formats: MP4, AVI, MOV, MKV, and other common video formats.",
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface

def main():
    """Launch the Gradio app."""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        show_error=True
    )

if __name__ == "__main__":
    main()