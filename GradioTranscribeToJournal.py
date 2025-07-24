#!/usr/bin/env python3

import os
import subprocess
import tempfile
import gradio as gr
import requests
import json
import difflib
import re
from pathlib import Path
from datetime import datetime, timezone
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Note: anthropic library not installed. Install with: pip install anthropic")

# Configuration
TEMP_DIR = "/tmp/video_transcribe"
FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
JOURNAL_DIR = "/Users/mini/Obsidian/AutumnsGarden/Journal/2025 Auto"
LM_STUDIO_ENDPOINT = "http://localhost:1234/v1/chat/completions"  # Fixed the double http://
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # Default Claude model
ANTHROPIC_API_KEY = ""  # Removed exposed API key - configure in secrets.json or environment variable
LOCAL_MODEL_TOKEN_LIMIT = 4096  # Token limit for local models before fallback

# Prompt Caching Configuration
ENABLE_PROMPT_CACHING = True  # Enable Anthropic prompt caching to reduce costs
CACHE_LIFETIME_HOURS = 1  # Cache lifetime in hours (1 or 5 minute default)


def estimate_tokens(text):
    """Rough estimate of token count (approximately 4 characters per token)."""
    return len(text) // 4


def test_model_connection(provider, lm_studio_endpoint, model_name):
    """Test the LLM model connection and return status."""
    if provider == "anthropic":
        return test_anthropic_connection()
    else:
        return test_local_connection(lm_studio_endpoint, model_name)


def test_anthropic_connection(anthropic_api_key=None):
    """Test Anthropic API connection."""
    try:
        if not ANTHROPIC_AVAILABLE:
            error_msg = "❌ **Anthropic Library Missing**\n\n**Issue:** anthropic library not installed\n**Solution:** Run: pip install anthropic"
            return False, error_msg
            
        # Use the API key from configuration if not provided
        api_key = anthropic_api_key or ANTHROPIC_API_KEY
        if not api_key or not api_key.strip():
            error_msg = "❌ **API Key Missing**\n\n**Issue:** Anthropic API key not set\n**Solution:** Set ANTHROPIC_API_KEY in the script configuration"
            return False, error_msg
            
        client = anthropic.Anthropic(api_key=api_key)
        
        # Simple test prompt
        test_prompt = "Please respond with exactly: 'Model test successful'"
        
        print(f"Testing connection to Anthropic API with model {ANTHROPIC_MODEL}")
        
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=20,
            temperature=0.1,
            messages=[
                {"role": "user", "content": test_prompt}
            ]
        )
        
        model_response = response.content[0].text.strip()
        
        success_msg = f"✅ **Connection Successful!**\n\n**Provider:** Anthropic\n**Model:** {ANTHROPIC_MODEL}\n**Response:** {model_response}\n**Status:** Ready for transcription"
        return True, success_msg
        
    except anthropic.AuthenticationError:
        error_msg = f"❌ **Authentication Failed**\n\n**Issue:** Invalid API key\n**Solution:** Check your Anthropic API key"
        return False, error_msg
        
    except anthropic.RateLimitError:
        error_msg = f"❌ **Rate Limit Exceeded**\n\n**Issue:** Too many requests\n**Solution:** Wait a moment and try again"
        return False, error_msg
        
    except Exception as e:
        error_msg = f"❌ **Anthropic API Error**\n\n**Error:** {str(e)}"
        return False, error_msg


def test_local_connection(lm_studio_endpoint, model_name):
    """Test local LM Studio connection."""
    try:
        headers = {"Content-Type": "application/json"}
        
        # Simple test prompt
        test_prompt = "Please respond with exactly: 'Model test successful'"
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": test_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 20,
        }
        
        print(f"Testing connection to {lm_studio_endpoint} with model {model_name}")
        
        response = requests.post(
            lm_studio_endpoint, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            model_response = result["choices"][0]["message"]["content"].strip()
            
            success_msg = f"✅ **Connection Successful!**\n\n**Provider:** Local (LM Studio)\n**Endpoint:** {lm_studio_endpoint}\n**Model:** {model_name}\n**Response:** {model_response}\n**Status:** Ready for transcription"
            return True, success_msg
            
        else:
            error_msg = f"❌ **HTTP Error {response.status_code}**\n\n**Endpoint:** {lm_studio_endpoint}\n**Model:** {model_name}\n**Error:** {response.text}"
            return False, error_msg
            
    except requests.exceptions.ConnectionError:
        error_msg = f"❌ **Connection Failed**\n\n**Endpoint:** {lm_studio_endpoint}\n**Issue:** Cannot connect to LM Studio\n**Solution:** Make sure LM Studio is running and the endpoint is correct"
        return False, error_msg
        
    except requests.exceptions.Timeout:
        error_msg = f"❌ **Timeout Error**\n\n**Endpoint:** {lm_studio_endpoint}\n**Issue:** Request timed out\n**Solution:** Check if model is loaded in LM Studio"
        return False, error_msg
        
    except KeyError as e:
        error_msg = f"❌ **Invalid Response**\n\n**Endpoint:** {lm_studio_endpoint}\n**Issue:** Unexpected API response format\n**Error:** {str(e)}"
        return False, error_msg
        
    except Exception as e:
        error_msg = f"❌ **Unknown Error**\n\n**Endpoint:** {lm_studio_endpoint}\n**Error:** {str(e)}"
        return False, error_msg


def ensure_directories():
    """Create temporary and journal directories if they don't exist."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(JOURNAL_DIR, exist_ok=True)


def get_file_metadata(file_path):
    """Get file creation and modification times."""
    try:
        stat = os.stat(file_path)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        return mod_time
    except:
        return datetime.now()

def get_video_recording_time(file_path):
    """Extract recording time from video metadata using ffprobe."""
    try:
        # Try to get creation time from video metadata
        cmd = [
            "ffprobe", 
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=creation_time:format_tags=creation_time",
            "-of", "csv=p=0",
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            # Parse the creation time from ffprobe output
            creation_time_str = result.stdout.strip().split('\n')[0]
            if creation_time_str and creation_time_str != "N/A":
                # Handle different datetime formats
                for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        dt = datetime.strptime(creation_time_str, fmt)
                        # If it has Z suffix, it's UTC - convert to local time
                        if creation_time_str.endswith('Z'):
                            dt = dt.replace(tzinfo=timezone.utc).astimezone()
                            # Remove timezone info to return naive datetime
                            dt = dt.replace(tzinfo=None)
                        return dt
                    except ValueError:
                        continue
        
        # Fallback to file modification time
        print(f"Using file modification time for {Path(file_path).name}")
        return get_file_metadata(file_path)
        
    except Exception as e:
        print(f"Error getting video metadata for {Path(file_path).name}: {e}")
        return get_file_metadata(file_path)

def parse_file_paths(input_text):
    """Parse file paths from input text, handling both space-separated and line-separated formats."""
    if not input_text or not input_text.strip():
        return []
    
    # First, try to split by newlines (original format)
    lines = [line.strip() for line in input_text.strip().split('\n') if line.strip()]
    
    # If we only have one line, it might be space-separated paths
    if len(lines) == 1:
        # Check if it contains multiple paths by looking for common video extensions
        single_line = lines[0]
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm', '.flv', '.wmv']
        
        # Count potential file paths by looking for extensions
        extension_count = sum(single_line.lower().count(ext) for ext in video_extensions)
        
        if extension_count > 1:
            # This looks like space-separated paths
            # Split by spaces but be careful about paths with spaces
            parts = single_line.split()
            paths = []
            current_path = ""
            
            for part in parts:
                if current_path:
                    current_path += " " + part
                else:
                    current_path = part
                
                # Check if this looks like a complete path (ends with video extension)
                if any(current_path.lower().endswith(ext) for ext in video_extensions):
                    paths.append(current_path)
                    current_path = ""
            
            # Add any remaining path
            if current_path:
                paths.append(current_path)
            
            return paths
    
    # Return the line-separated format
    return lines


def convert_to_mp3(video_path, progress_callback=None):
    """Convert video file to MP3 format."""
    video_name = Path(video_path).stem
    mp3_path = os.path.join(TEMP_DIR, f"{video_name}.mp3")

    if progress_callback:
        progress_callback("Converting to MP3...")

    cmd = [FFMPEG_PATH, "-y", "-i", video_path, "-q:a", "0", "-map", "a", mp3_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return mp3_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error converting video to MP3: {e.stderr}")


def transcribe_audio(mp3_path, progress_callback=None):
    """Transcribe MP3 file using parakeet-mlx."""
    if progress_callback:
        progress_callback("Transcribing audio...")
        
    try:
        cmd = [
            "parakeet-mlx",
            "--output-format",
            "txt",
            "--output-dir",
            TEMP_DIR,
            mp3_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        txt_path = os.path.join(TEMP_DIR, f"{Path(mp3_path).stem}.txt")

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                transcript = f.read()
            os.remove(txt_path)
            return transcript
        else:
            raise Exception("Transcription file was not created")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during transcription: {e.stderr}")


def generate_brief_title(transcript, provider, lm_studio_endpoint=None, model_name=None, progress_callback=None):
    """Generate a brief title (8 words max) for the transcript using LLM."""
    if progress_callback:
        progress_callback("Generating title...")
        
    try:
        # Clean up transcript for title generation
        clean_transcript = transcript.strip()[:1000]  # First 1000 chars
        
        title_prompt = """Generate a brief, descriptive title for this transcript. Requirements:
- Maximum 8 words
- Capture the main topic or theme
- Be specific and informative
- No quotation marks or special formatting
- Just return the title, nothing else

Transcript excerpt:"""

        # Check token limit for local provider and fallback if needed
        if provider == "local":
            estimated_tokens = estimate_tokens(clean_transcript + title_prompt)
            if estimated_tokens > LOCAL_MODEL_TOKEN_LIMIT:
                print(f"Token limit exceeded for title generation, falling back to Anthropic")
                if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
                    return generate_title_anthropic(clean_transcript, title_prompt)
                else:
                    # Fallback to simple extraction
                    words = [w for w in transcript.split()[:10] if len(w) > 2][:8]
                    return " ".join(words) if words else "Video Transcript"

        if provider == "anthropic":
            return generate_title_anthropic(clean_transcript, title_prompt)
        else:
            return generate_title_local(clean_transcript, title_prompt, lm_studio_endpoint, model_name)
            
    except Exception as e:
        print(f"Title generation failed: {e}")
        # Fallback: extract first few meaningful words from transcript
        words = [w for w in transcript.split()[:10] if len(w) > 2][:8]
        return " ".join(words) if words else "Video Transcript"


def generate_title_anthropic(clean_transcript, title_prompt):
    """Generate title using Anthropic API with prompt caching for cost optimization."""
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        raise Exception("Anthropic API not available or key not set")
        
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Use prompt caching if enabled - separate cached prompt from variable content
    if ENABLE_PROMPT_CACHING:
        print(f"Using prompt caching for title generation request")
        
        # Configure cache headers if extended cache lifetime is desired
        extra_headers = {}
        if CACHE_LIFETIME_HOURS > 1:
            extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=20,
            temperature=0.65,
            extra_headers=extra_headers,
            system=[
                {
                    "type": "text",
                    "text": title_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user", 
                    "content": f"Generate a title for this transcript excerpt:\n\n{clean_transcript}"
                }
            ]
        )
    else:
        # Fallback to non-cached version
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=20,
            temperature=0.65,
            messages=[
                {"role": "user", "content": f"{title_prompt}\n\n{clean_transcript}"}
            ]
        )
    
    title = response.content[0].text.strip()
    # Clean and limit the title
    title = re.sub(r'[^\w\s-]', '', title)  # Remove special chars except hyphens
    words = title.split()[:8]  # Limit to 8 words
    return " ".join(words)


def generate_title_local(clean_transcript, title_prompt, lm_studio_endpoint, model_name):
    """Generate title using local LM Studio."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": title_prompt},
            {"role": "user", "content": clean_transcript},
        ],
        "temperature": 0.65,
        "max_tokens": 100,
    }
    
    response = requests.post(
        lm_studio_endpoint, headers=headers, json=payload, timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        title = result["choices"][0]["message"]["content"].strip()
        # Clean and limit the title
        title = re.sub(r'[^\w\s-]', '', title)  # Remove special chars except hyphens
        words = title.split()[:8]  # Limit to 8 words
        return " ".join(words)
    else:
        # Fallback: extract first few meaningful words
        words = clean_transcript.split()[:8]
        return " ".join(words)


def enhance_transcript_with_llm(
    transcript, enhancement_prompt, provider, lm_studio_endpoint=None, model_name=None, progress_callback=None
):
    """Enhance transcript using LLM (either local or Anthropic)."""
    if progress_callback:
        progress_callback("Enhancing with LLM...")
        
    print(f"\n=== LLM Enhancement Debug ===")
    print(f"Provider: {provider}")
    if provider == "local":
        print(f"Endpoint: {lm_studio_endpoint}")
        print(f"Model: {model_name}")
    else:
        print(f"Model: {ANTHROPIC_MODEL}")
    print(f"Transcript length: {len(transcript)} chars")
    print(f"Prompt length: {len(enhancement_prompt)} chars")
    print(f"First 100 chars of transcript: {transcript[:100]}...")
    
    try:
        # Check if transcript is too short
        word_count = len(transcript.split())
        print(f"Word count: {word_count}")
        
        if word_count < 10:
            print("Transcript too short, skipping LLM enhancement")
            return transcript, "Transcript too short for enhancement"
        
        # Check token limit for local provider
        if provider == "local":
            estimated_tokens = estimate_tokens(transcript + enhancement_prompt)
            print(f"Estimated tokens: {estimated_tokens}")
            
            if estimated_tokens > LOCAL_MODEL_TOKEN_LIMIT:
                print(f"Token limit exceeded ({estimated_tokens} > {LOCAL_MODEL_TOKEN_LIMIT}), falling back to Anthropic")
                if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
                    if progress_callback:
                        progress_callback("Token limit exceeded, using Anthropic fallback...")
                    return enhance_transcript_anthropic(transcript, enhancement_prompt)
                else:
                    return None, f"Token limit exceeded ({estimated_tokens} tokens) and Anthropic fallback not available"
        
        if provider == "anthropic":
            return enhance_transcript_anthropic(transcript, enhancement_prompt)
        else:
            return enhance_transcript_local(transcript, enhancement_prompt, lm_studio_endpoint, model_name)
            
    except Exception as e:
        error_msg = f"Enhancement Error: {str(e)}"
        print(f"General Error: {error_msg}")
        return None, error_msg


def enhance_transcript_anthropic(transcript, enhancement_prompt):
    """Enhance transcript using Anthropic API with prompt caching for cost optimization."""
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        raise Exception("Anthropic API not available or key not set")
        
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    print(f"Sending request to Anthropic...")
    
    # Use prompt caching if enabled - separate cached prompt from variable content
    if ENABLE_PROMPT_CACHING:
        print(f"Using prompt caching for enhancement request")
        
        # Configure cache headers if extended cache lifetime is desired
        extra_headers = {}
        if CACHE_LIFETIME_HOURS > 1:
            extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=8000,
            temperature=0.6,
            extra_headers=extra_headers,
            system=[
                {
                    "type": "text",
                    "text": enhancement_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user", 
                    "content": f"Please enhance this transcript:\n\n{transcript}"
                }
            ]
        )
    else:
        # Fallback to non-cached version
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=8000,
            temperature=0.6,
            messages=[
                {"role": "user", "content": f"{enhancement_prompt}\n\n{transcript}"}
            ]
        )
    
    enhanced_text = response.content[0].text
    
    # Check if LLM returned a SKIP response
    if enhanced_text.strip().startswith("SKIP:"):
        print(f"LLM skipped transcript: {enhanced_text}")
        return transcript, enhanced_text
    
    print(f"Enhancement successful, enhanced text length: {len(enhanced_text)} chars")
    return enhanced_text.strip(), None


def enhance_transcript_local(transcript, enhancement_prompt, lm_studio_endpoint, model_name):
    """Enhance transcript using local LM Studio."""
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": enhancement_prompt},
            {"role": "user", "content": transcript},
        ],
        "temperature": 0.6,
        "max_tokens": 4000,
    }
    
    print(f"Sending request to LLM...")
    response = requests.post(
        lm_studio_endpoint, headers=headers, json=payload, timeout=120
    )
    print(f"Response status: {response.status_code}")
    response.raise_for_status()

    result = response.json()
    enhanced_text = result["choices"][0]["message"]["content"]
    
    # Check if LLM returned a SKIP response
    if enhanced_text.strip().startswith("SKIP:"):
        print(f"LLM skipped transcript: {enhanced_text}")
        return transcript, enhanced_text
    
    print(f"Enhancement successful, enhanced text length: {len(enhanced_text)} chars")
    return enhanced_text.strip(), None


def create_text_diff(original, enhanced):
    """Create a unified diff between original and enhanced text."""
    original_lines = original.splitlines(keepends=True)
    enhanced_lines = enhanced.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            original_lines,
            enhanced_lines,
            fromfile="Original Transcript",
            tofile="Enhanced Transcript",
            lineterm="",
        )
    )

    return "".join(diff)


def create_journal_entry(video_name, file_timestamp, transcript, brief_title=""):
    """Create a properly formatted header-based journal entry."""
    date_str = file_timestamp.strftime("%Y-%m-%d")
    time_str = file_timestamp.strftime("%H:%M:%S")
    
    # Use brief title if available, otherwise use video name
    display_title = brief_title if brief_title else video_name
    
    entry = f"""## {display_title} ({date_str} {time_str})
### Video: {video_name}

{transcript}

---

"""
    return entry


def append_to_journal(journal_entry, file_timestamp, custom_path=None, progress_callback=None):
    """Append the journal entry to the appropriate daily journal file."""
    if progress_callback:
        progress_callback("Saving to journal...")
        
    if custom_path:
        journal_path = custom_path
    else:
        date_str = file_timestamp.strftime("%Y-%m-%d")
        journal_path = os.path.join(JOURNAL_DIR, f"{date_str}-auto.md")

    if not os.path.exists(journal_path):
        header = f"# Daily Auto Journal - {file_timestamp.strftime('%B %d, %Y')}\n\n"
        with open(journal_path, "w") as f:
            f.write(header)

    with open(journal_path, "a") as f:
        f.write(journal_entry)

    return journal_path


def process_single_video(
    video_file,
    auto_save,
    custom_journal_path,
    enhance_with_llm,
    enhancement_prompt,
    provider,
    lm_studio_endpoint,
    model_name,
    progress_callback=None,
):
    """Process a single video file and return formatted transcript."""
    video_path = video_file.name
    video_name = Path(video_path).stem
    
    # Use recording time if available from file object, otherwise get from metadata
    if hasattr(video_file, 'recording_time'):
        file_timestamp = video_file.recording_time
        print(f"Using pre-determined recording time: {file_timestamp}")
    else:
        file_timestamp = get_video_recording_time(video_path)
        print(f"Getting recording time from metadata: {file_timestamp}")
    
    print(f"\n=== Processing {video_name} ===")
    print(f"Enhance with LLM: {enhance_with_llm}")
    print(f"Has enhancement prompt: {bool(enhancement_prompt.strip())}")

    try:
        # Convert video to MP3
        mp3_path = convert_to_mp3(video_path, progress_callback)

        try:
            # Transcribe audio
            original_transcript = transcribe_audio(mp3_path, progress_callback)
            print(f"Transcription complete: {len(original_transcript)} chars")

            # Initialize variables
            final_transcript = original_transcript
            brief_title = ""
            diff_output = ""
            enhancement_status = ""

            # Generate title and enhance with LLM if enabled
            if enhance_with_llm and enhancement_prompt.strip():
                print("Starting LLM enhancement...")
                
                # Generate brief title first
                brief_title = generate_brief_title(
                    original_transcript, provider, lm_studio_endpoint, model_name, progress_callback
                )
                
                # Then enhance transcript
                enhanced_transcript, error = enhance_transcript_with_llm(
                    original_transcript,
                    enhancement_prompt,
                    provider,
                    lm_studio_endpoint,
                    model_name,
                    progress_callback,
                )

                if enhanced_transcript and enhanced_transcript != original_transcript:
                    final_transcript = enhanced_transcript
                    diff_output = create_text_diff(
                        original_transcript, enhanced_transcript
                    )
                    enhancement_status = "✅ Enhanced with LLM"
                    print("Enhancement successful")
                elif enhanced_transcript == original_transcript:
                    enhancement_status = "⚠️ No changes made by LLM"
                    print("LLM made no changes")
                else:
                    enhancement_status = f"❌ Enhancement failed: {error}"
                    print(f"Enhancement failed: {error}")
            elif enhance_with_llm:
                enhancement_status = "⚠️ No enhancement prompt provided"
                print("No enhancement prompt provided")
            else:
                enhancement_status = "⏭️ LLM enhancement disabled"
                print("LLM enhancement is disabled")

            # Create journal entry with final transcript
            journal_entry = create_journal_entry(
                video_name, file_timestamp, final_transcript, brief_title
            )

            # Auto-save to journal if enabled
            journal_path = None
            if auto_save:
                journal_path = append_to_journal(
                    journal_entry,
                    file_timestamp,
                    custom_journal_path if custom_journal_path.strip() else None,
                    progress_callback,
                )

            # Format the display transcript
            time_str = file_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            display_title = brief_title if brief_title else video_name
            
            display_transcript = f"""# {display_title}
**Video:** {video_name}
**File Date:** {time_str}
{f"**Saved to:** {journal_path}" if journal_path else ""}
**Enhancement:** {enhancement_status}

{final_transcript}

---

"""
            return display_transcript, journal_entry, diff_output, None

        finally:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

    except Exception as e:
        error_msg = f"""# {video_name}
**File Date:** {file_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Error:** {str(e)}

---

"""
        return error_msg, "", "", str(e)


def process_videos(
    file_paths_input,
    video_files,
    auto_save,
    custom_journal_path,
    enhance_with_llm,
    enhancement_prompt,
    provider,
    lm_studio_endpoint,
    model_name,
    enable_caching,
    progress=gr.Progress(),
):
    """Main processing function for multiple video files."""
    global ENABLE_PROMPT_CACHING
    
    # Temporarily override global caching setting with UI setting
    original_caching_setting = ENABLE_PROMPT_CACHING
    ENABLE_PROMPT_CACHING = enable_caching
    
    try:
        print(f"\n=== Starting batch processing ===")
        print(f"Prompt caching: {'enabled' if enable_caching else 'disabled'}")
        print(f"LLM Enhancement enabled: {enhance_with_llm}")
        print(f"Provider: {provider}")
        if provider == "local":
            print(f"LM Studio endpoint: {lm_studio_endpoint}")
            print(f"Model: {model_name}")
        else:
            print(f"Model: {ANTHROPIC_MODEL}")

        # Prepare file list from either file paths or uploaded files
        files_to_process = []

        # Handle file paths input
        if file_paths_input and file_paths_input.strip():
            progress(0.0, desc="Parsing file paths...")
            file_paths = parse_file_paths(file_paths_input)
            
            print(f"Parsed {len(file_paths)} file paths:")
            for path in file_paths:
                print(f"  - {path}")
            
            # Check all files exist first
            missing_files = []
            valid_files = []
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    valid_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            if missing_files:
                return f"Files not found:\n" + "\n".join(f"- {path}" for path in missing_files), "", ""
            
            # Get recording times and sort by oldest to newest
            progress(0.02, desc="Reading video metadata for chronological sorting...")
            files_with_times = []
            
            for file_path in valid_files:
                recording_time = get_video_recording_time(file_path)
                files_with_times.append((file_path, recording_time))
                print(f"File: {Path(file_path).name} - Recording time: {recording_time}")
            
            # Sort by recording time (oldest first)
            files_with_times.sort(key=lambda x: x[1])
            
            print("\nProcessing order (oldest to newest):")
            for i, (file_path, recording_time) in enumerate(files_with_times, 1):
                print(f"  {i}. {Path(file_path).name} ({recording_time})")
            
            # Create mock file objects for compatibility
            class MockFile:
                def __init__(self, path, recording_time):
                    self.name = path
                    self.recording_time = recording_time

            files_to_process = [MockFile(path, time) for path, time in files_with_times]

        # Handle uploaded files
        elif video_files and len(video_files) > 0:
            progress(0.0, desc="Preparing uploaded files...")
            
            # For uploaded files, also sort by recording time
            progress(0.02, desc="Reading metadata for uploaded files...")
            files_with_times = []
            
            for video_file in video_files:
                recording_time = get_video_recording_time(video_file.name)
                files_with_times.append((video_file, recording_time))
                print(f"Uploaded file: {Path(video_file.name).name} - Recording time: {recording_time}")
            
            # Sort by recording time (oldest first)
            files_with_times.sort(key=lambda x: x[1])
            
            print("\nProcessing order for uploaded files (oldest to newest):")
            for i, (video_file, recording_time) in enumerate(files_with_times, 1):
                print(f"  {i}. {Path(video_file.name).name} ({recording_time})")
            
            files_to_process = [video_file for video_file, _ in files_with_times]

        if not files_to_process:
            return "Please provide video files via file paths or upload.", "", ""

        ensure_directories()

        all_transcripts = []
        all_journal_entries = []
        all_diffs = []
        total_files = len(files_to_process)

        # Add header with summary
        progress(0.05, desc="Creating summary header...")
        summary = f"""# Video Transcription Results
**Total Files:** {total_files}
**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Auto-save:** {'Enabled' if auto_save else 'Disabled'}
**LLM Enhancement:** {'Enabled' if enhance_with_llm else 'Disabled'}
**Input Method:** {'File Paths' if file_paths_input and file_paths_input.strip() else 'File Upload'}

"""
        all_transcripts.append(summary)

        # Process each file with detailed progress tracking
        for i, video_file in enumerate(files_to_process):
            file_progress_start = 0.1 + (i * 0.8 / total_files)
            file_progress_end = 0.1 + ((i + 1) * 0.8 / total_files)
            
            video_name = Path(video_file.name).stem
            
            # Create a progress callback for this specific file
            def file_progress_callback(step_description):
                current_progress = file_progress_start + (file_progress_end - file_progress_start) * 0.5
                progress(current_progress, desc=f"[{i+1}/{total_files}] {video_name}: {step_description}")

            progress(file_progress_start, desc=f"[{i+1}/{total_files}] Starting {video_name}")

            transcript, journal_entry, diff_output, error = process_single_video(
                video_file,
                auto_save,
                custom_journal_path,
                enhance_with_llm,
                enhancement_prompt,
                provider,
                lm_studio_endpoint,
                model_name,
                file_progress_callback,
            )

            all_transcripts.append(transcript)
            if journal_entry:
                all_journal_entries.append(journal_entry)
            if diff_output:
                all_diffs.append(
                    f"## {Path(video_file.name).stem}\n\n```diff\n{diff_output}\n```\n\n"
                )

            progress(file_progress_end, desc=f"[{i+1}/{total_files}] Completed {video_name}")

            # Clean up uploaded file copies only (not files from paths)
            if video_files and len(video_files) > 0:  # Only if using uploaded files
                try:
                    if os.path.exists(video_file.name) and "/tmp/" in video_file.name:
                        os.remove(video_file.name)
                except:
                    pass

        progress(1.0, desc="All files processed successfully!")
        return "".join(all_transcripts), "".join(all_journal_entries), "".join(all_diffs)
    
    finally:
        # Always restore original caching setting
        ENABLE_PROMPT_CACHING = original_caching_setting


def create_interface():
    """Create and configure the Gradio interface."""

    default_prompt = """You are an expert transcript editor. Your task is to enhance a raw transcript while preserving authenticity.

## Pre-Processing Check
First, determine if the transcript should be processed:
- If less than 50 meaningful words, return only: SKIP: Too short
- If mostly silence/non-verbal, return only: SKIP: No meaningful content
- If corrupted/incoherent, return only: SKIP: Corrupted transcript

## Processing Rules

1. **Preserve ALL original words** (except filler removal)
2. **Add structure and formatting:**
   - Insert paragraph breaks between distinct thoughts
   - Add topic headers: ### *Topic: [Name]*
   - Add "circling back" headers: ### *Circling Back: [Name]*
   - Add brief context notes: ***[context: description]*** (max 10 words)

3. **Remove only:**
   - Filler words: um, uh, like (unless meaningful)
   - Fix obvious transcription errors

4. **Never:**
   - Summarize or condense
   - Rewrite for style
   - Remove repetitions or tangents

## Output
Return the enhanced transcript with all formatting applied. Assume single speaker unless absolutely clear otherwise.

Remember: Be conservative with changes. When in doubt, preserve the original."""

    with gr.Blocks(
        theme=gr.themes.Base(), title="Video Transcription Tool"
    ) as interface:
        gr.Markdown("# Video Transcription Tool")
        gr.Markdown(
            "Upload video files to generate transcripts with optional LLM enhancement and automatic journal integration."
        )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("File Paths"):
                        file_paths_input = gr.Textbox(
                            label="Video File Paths",
                            lines=8,
                            placeholder="Paste file paths here:\n\nOne per line:\n/path/to/video1.mp4\n/path/to/video2.mp4\n\nOr space-separated:\n/path/to/video1.mp4 /path/to/video2.mp4 /path/to/video3.mp4",
                            info="Enter full paths to your video files (supports both line-separated and space-separated formats)",
                        )

                    with gr.TabItem("Upload Files"):
                        video_input = gr.File(
                            file_types=["video"],
                            label="Upload Video Files",
                            file_count="multiple",
                            height=200,
                        )

            with gr.Column(scale=1):
                auto_save = gr.Checkbox(value=True, label="Auto-save to Journal")

                custom_journal_path = gr.Textbox(
                    label="Custom Journal Path (optional)",
                    placeholder="/path/to/your/journal.md",
                )

        with gr.Accordion("LLM Settings", open=False):
            enhance_with_llm = gr.Checkbox(
                value=True,
                label="Enhance transcripts with LLM",
                info="Improve readability, add context, and generate titles",
            )

            provider = gr.Radio(
                choices=["local", "anthropic"],
                value="anthropic",
                label="LLM Provider",
                info="Choose between local LM Studio or Anthropic API"
            )

            with gr.Group(visible=False) as local_settings:
                gr.Markdown("### Local LM Studio Settings")
                with gr.Row():
                    lm_studio_endpoint = gr.Textbox(
                        value=LM_STUDIO_ENDPOINT,
                        label="LM Studio API Endpoint",
                        info="Usually http://localhost:1234/v1/chat/completions",
                    )

                    model_name = gr.Textbox(
                        value="qwen/qwen3-30b-a3b",
                        label="Model Name",
                        info="Model identifier in LM Studio",
                    )

            with gr.Group(visible=True) as anthropic_settings:
                gr.Markdown("### Anthropic API Settings")
                gr.Markdown(f"**Model:** {ANTHROPIC_MODEL}")
                if ANTHROPIC_API_KEY:
                    gr.Markdown("✅ **API Key:** Configured in script")
                else:
                    gr.Markdown("❌ **API Key:** Not set - configure ANTHROPIC_API_KEY in script")
                
                # Prompt caching configuration
                enable_caching = gr.Checkbox(
                    value=ENABLE_PROMPT_CACHING,
                    label="Enable Prompt Caching",
                    info="Cache prompts to reduce API costs by up to 90% for repeated requests"
                )
                
                if ENABLE_PROMPT_CACHING:
                    gr.Markdown("💰 **Cost Optimization:** Caching enabled - prompts cached for reuse, significant savings expected")

            # Test Model Button and Output
            with gr.Row():
                test_model_btn = gr.Button("🧪 Test Model Connection", variant="secondary", size="sm")
                
            test_output = gr.Markdown(
                label="Model Test Results",
                value="",
                visible=False
            )

            enhancement_prompt = gr.Textbox(
                value=default_prompt,
                label="Enhancement Prompt",
                lines=6,
                info="Instructions for the LLM on how to improve the transcript",
            )

        with gr.Row():
            process_btn = gr.Button("Process Videos", variant="primary", size="lg")

        with gr.Tabs():
            with gr.TabItem("Transcripts"):
                transcript_output = gr.Textbox(
                    label="Final Transcripts",
                    lines=20,
                    max_lines=50,
                    show_copy_button=True,
                )

            with gr.TabItem("Journal Entries"):
                journal_output = gr.Textbox(
                    label="Journal Entries (Header Format)",
                    lines=20,
                    max_lines=50,
                    show_copy_button=True,
                )

            with gr.TabItem("Diffs"):
                diff_output = gr.Textbox(
                    label="Changes Made by LLM",
                    lines=20,
                    max_lines=50,
                    show_copy_button=True,
                    placeholder="Diffs will appear here when LLM enhancement is enabled...",
                )

        # Provider change handler
        def update_provider_settings(provider_choice):
            if provider_choice == "anthropic":
                return {
                    local_settings: gr.Group(visible=False),
                    anthropic_settings: gr.Group(visible=True)
                }
            else:
                return {
                    local_settings: gr.Group(visible=True),
                    anthropic_settings: gr.Group(visible=False)
                }

        provider.change(
            fn=update_provider_settings,
            inputs=[provider],
            outputs=[local_settings, anthropic_settings]
        )

        # Test model connection function
        def test_model_and_show_result(provider_choice, endpoint, model):
            success, message = test_model_connection(provider_choice, endpoint, model)
            
            # Show/hide and update the test output
            return {
                test_output: gr.Markdown(value=message, visible=True)
            }

        # Set up the test button
        test_model_btn.click(
            fn=test_model_and_show_result,
            inputs=[provider, lm_studio_endpoint, model_name],
            outputs=[test_output]
        )

        # Set up the processing function
        process_btn.click(
            fn=process_videos,
            inputs=[
                file_paths_input,
                video_input,
                auto_save,
                custom_journal_path,
                enhance_with_llm,
                enhancement_prompt,
                provider,
                lm_studio_endpoint,
                model_name,
                enable_caching,
            ],
            outputs=[transcript_output, journal_output, diff_output],
        )

        gr.Markdown(
            """
        ### Features:
        - **Video Transcription**: Converts videos to text using parakeet-mlx
        - **Dual LLM Support**: Choose between local LM Studio or Anthropic Claude Sonnet
        - **LLM Enhancement**: Improves readability and generates titles using your chosen provider
        - **Prompt Caching**: Anthropic prompt caching reduces API costs by up to 90% for repeated requests
        - **Model Testing**: Test your LLM connection before processing videos
        - **Header Format**: Clean header-based format instead of callout blocks for better Obsidian navigation
        - **Brief Titles**: Auto-generated descriptive titles (8 words max) for each transcript
        - **Enhanced Progress**: Detailed progress tracking for each processing step
        - **Diff Viewing**: See exactly what changes the LLM made
        - **Auto-Journal**: Automatically saves enhanced transcripts to daily journals
        - **Batch Processing**: Handle multiple videos at once
        - **Flexible Input**: Use file paths OR drag & drop upload  
        - **Smart Path Parsing**: Supports both line-separated and space-separated file paths
        - **Chronological Processing**: Automatically sorts videos by recording time (oldest to newest)
        
        ### Provider Setup:
        - **Local**: Make sure LM Studio is running at the specified endpoint with a model loaded
        - **Anthropic**: Set your API key in the ANTHROPIC_API_KEY variable at the top of the script
        - Use the "Test Model Connection" button to verify your setup
        
        ### Debug Info:
        - Check console output for detailed debug logs
        - LLM Enhancement is enabled by default and includes title generation
        - Claude Sonnet is excellent for transcript enhancement and structuring
        """
        )

    return interface


def main():
    """Launch the Gradio app."""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0", server_port=7868, share=False, show_error=True
    )


if __name__ == "__main__":
    main()