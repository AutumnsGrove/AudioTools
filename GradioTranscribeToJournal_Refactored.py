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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Note: anthropic library not installed. Install with: pip install anthropic")

# Load secrets from file
def load_secrets() -> Dict[str, str]:
    """Load API keys from secrets.json file."""
    secrets_path = os.path.join(os.path.dirname(__file__), "secrets.json")
    try:
        with open(secrets_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: secrets.json not found at {secrets_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in secrets.json")
        return {}

# Configuration Management
@dataclass
class AppConfig:
    """Centralized application configuration."""
    # Paths
    temp_dir: str = "/tmp/video_transcribe"
    ffmpeg_path: str = "/opt/homebrew/bin/ffmpeg"
    journal_dir: str = "/Users/mini/Obsidian/AutumnsGarden/Journal/2025 Auto"
    
    # LLM Settings
    local_endpoint: str = "http://localhost:1234/v1/chat/completions"
    local_model: str = "qwen/qwen3-30b-a3b"
    local_token_limit: int = 4096
    
    # API Keys
    secrets: Dict[str, str] = field(default_factory=load_secrets)
    
    # Token estimation
    chars_per_token: int = 4
    
    # Title generation
    title_max_words: int = 8
    title_max_tokens: int = 100
    title_temperature: float = 0.65
    
    # Enhancement settings
    enhancement_max_tokens: int = 8000
    enhancement_temperature: float = 0.6
    
    # OpenRouter Provider Routing
    openrouter_providers: List[str] = field(default_factory=lambda: [
        "anthropic",  # High precision, no data collection
        "openai",     # High precision, minimal data collection
        "google",     # High precision, limited data collection
    ])
    
    # Provider routing parameters
    provider_routing: Dict[str, Any] = field(default_factory=lambda: {
        "require_parameters": True,
        "data_collection": "deny",
        "quantization": "fp8",  # fp8 and higher precision only
    })

class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LOCAL = "local"

@dataclass
class ModelConfig:
    """Model configuration for different providers."""
    provider: LLMProvider
    model_id: str
    max_tokens: int
    temperature: float
    endpoint: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

class LLMClient:
    """Unified LLM client supporting multiple providers."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.models = self._setup_models()
    
    def _setup_models(self) -> Dict[LLMProvider, ModelConfig]:
        """Setup model configurations for all providers."""
        return {
            LLMProvider.ANTHROPIC: ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_id="claude-sonnet-4-20250514",
                max_tokens=8000,
                temperature=0.6
            ),
            LLMProvider.OPENROUTER: ModelConfig(
                provider=LLMProvider.OPENROUTER,
                model_id="moonshotai/kimi-k2:floor",
                max_tokens=8000,
                temperature=0.6,
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                headers=self._get_openrouter_headers()
            ),
            LLMProvider.LOCAL: ModelConfig(
                provider=LLMProvider.LOCAL,
                model_id=self.config.local_model,
                max_tokens=4000,
                temperature=0.6,
                endpoint=self.config.local_endpoint
            )
        }
    
    def _get_openrouter_headers(self) -> Dict[str, str]:
        """Get OpenRouter headers with provider routing."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.secrets.get('openrouter_api_key', '')}",
            "HTTP-Referer": "https://github.com/user/AudioTools",
            "X-Title": "AudioTools Transcription"
        }
        
        # Add provider routing for privacy and precision
        if self.config.provider_routing:
            headers["OpenRouter-Provider-Routing"] = json.dumps({
                "providers": self.config.openrouter_providers,
                "require_parameters": self.config.provider_routing["require_parameters"],
                "data_collection": self.config.provider_routing["data_collection"],
                "quantization": self.config.provider_routing["quantization"]
            })
        
        return headers
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.config.chars_per_token
    
    def test_connection(self, provider: LLMProvider) -> Tuple[bool, str]:
        """Test connection to specified provider."""
        try:
            test_prompt = "Please respond with exactly: 'Model test successful'"
            
            if provider == LLMProvider.ANTHROPIC:
                return self._test_anthropic(test_prompt)
            elif provider == LLMProvider.OPENROUTER:
                return self._test_openrouter(test_prompt)
            elif provider == LLMProvider.LOCAL:
                return self._test_local(test_prompt)
            else:
                return False, f"❌ **Unknown Provider**: {provider}"
                
        except Exception as e:
            return False, f"❌ **Connection Error**: {str(e)}"
    
    def _test_anthropic(self, prompt: str) -> Tuple[bool, str]:
        """Test Anthropic connection."""
        if not ANTHROPIC_AVAILABLE:
            return False, "❌ **Anthropic Library Missing**\\n\\nInstall with: pip install anthropic"
        
        api_key = self.config.secrets.get("anthropic_api_key", "")
        if not api_key:
            return False, "❌ **API Key Missing**\\n\\nAdd anthropic_api_key to secrets.json"
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            model = self.models[LLMProvider.ANTHROPIC]
            
            response = client.messages.create(
                model=model.model_id,
                max_tokens=20,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.content[0].text.strip()
            return True, f"✅ **Connection Successful!**\\n\\n**Provider:** Anthropic\\n**Model:** {model.model_id}\\n**Response:** {result}"
            
        except anthropic.AuthenticationError:
            return False, "❌ **Authentication Failed**\\n\\nCheck your Anthropic API key"
        except anthropic.RateLimitError:
            return False, "❌ **Rate Limit Exceeded**\\n\\nWait a moment and try again"
        except Exception as e:
            return False, f"❌ **Anthropic API Error**\\n\\n{str(e)}"
    
    def _test_openrouter(self, prompt: str) -> Tuple[bool, str]:
        """Test OpenRouter connection."""
        api_key = self.config.secrets.get("openrouter_api_key", "")
        if not api_key:
            return False, "❌ **API Key Missing**\\n\\nAdd openrouter_api_key to secrets.json"
        
        try:
            model = self.models[LLMProvider.OPENROUTER]
            payload = {
                "model": model.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 20,
                **self.config.provider_routing
            }
            
            response = requests.post(
                model.endpoint,
                headers=model.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip()
                return True, f"✅ **Connection Successful!**\\n\\n**Provider:** OpenRouter\\n**Model:** {model.model_id}\\n**Response:** {result}"
            else:
                return False, f"❌ **HTTP Error {response.status_code}**\\n\\n{response.text}"
                
        except requests.exceptions.RequestException as e:
            return False, f"❌ **Connection Error**\\n\\n{str(e)}"
        except Exception as e:
            return False, f"❌ **OpenRouter Error**\\n\\n{str(e)}"
    
    def _test_local(self, prompt: str) -> Tuple[bool, str]:
        """Test local LM Studio connection."""
        try:
            model = self.models[LLMProvider.LOCAL]
            payload = {
                "model": model.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 20,
            }
            
            response = requests.post(
                model.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"].strip()
                return True, f"✅ **Connection Successful!**\\n\\n**Provider:** Local\\n**Model:** {model.model_id}\\n**Response:** {result}"
            else:
                return False, f"❌ **HTTP Error {response.status_code}**\\n\\n{response.text}"
                
        except requests.exceptions.RequestException as e:
            return False, f"❌ **Connection Error**\\n\\nMake sure LM Studio is running"
        except Exception as e:
            return False, f"❌ **Local Error**\\n\\n{str(e)}"
    
    def generate_title(self, transcript: str, provider: LLMProvider) -> str:
        """Generate a brief title for the transcript."""
        try:
            clean_transcript = transcript.strip()[:1000]
            
            prompt = f"""Generate a brief, descriptive title for this transcript. Requirements:
- Maximum {self.config.title_max_words} words
- Capture the main topic or theme
- Be specific and informative
- No quotation marks or special formatting
- Just return the title, nothing else

Transcript excerpt: {clean_transcript}"""

            # Check token limit for local provider
            if provider == LLMProvider.LOCAL:
                estimated_tokens = self.estimate_tokens(clean_transcript + prompt)
                if estimated_tokens > self.config.local_token_limit:
                    print("Token limit exceeded for title generation, falling back to Anthropic")
                    if ANTHROPIC_AVAILABLE and self.config.secrets.get("anthropic_api_key"):
                        provider = LLMProvider.ANTHROPIC
                    else:
                        return self._fallback_title(transcript)
            
            response = self._make_request(provider, prompt, self.config.title_max_tokens, self.config.title_temperature)
            
            if response:
                # Clean and limit the title
                title = re.sub(r'[^\\w\\s-]', '', response)
                words = title.split()[:self.config.title_max_words]
                return " ".join(words)
            
            return self._fallback_title(transcript)
            
        except Exception as e:
            print(f"Title generation failed: {e}")
            return self._fallback_title(transcript)
    
    def _fallback_title(self, transcript: str) -> str:
        """Generate fallback title from transcript."""
        words = [w for w in transcript.split()[:10] if len(w) > 2][:self.config.title_max_words]
        return " ".join(words) if words else "Video Transcript"
    
    def enhance_transcript(self, transcript: str, prompt: str, provider: LLMProvider) -> Tuple[Optional[str], Optional[str]]:
        """Enhance transcript using specified provider."""
        try:
            # Check if transcript is too short
            word_count = len(transcript.split())
            if word_count < 10:
                return transcript, "Transcript too short for enhancement"
            
            # Check token limit for local provider
            if provider == LLMProvider.LOCAL:
                estimated_tokens = self.estimate_tokens(transcript + prompt)
                if estimated_tokens > self.config.local_token_limit:
                    print("Token limit exceeded, falling back to Anthropic")
                    if ANTHROPIC_AVAILABLE and self.config.secrets.get("anthropic_api_key"):
                        provider = LLMProvider.ANTHROPIC
                    else:
                        return None, f"Token limit exceeded ({estimated_tokens} tokens)"
            
            full_prompt = f"{prompt}\\n\\n{transcript}"
            response = self._make_request(provider, full_prompt, self.config.enhancement_max_tokens, self.config.enhancement_temperature)
            
            if response:
                # Check for SKIP response
                if response.strip().startswith("SKIP:"):
                    return transcript, response
                return response.strip(), None
            
            return None, "No response received"
            
        except Exception as e:
            return None, f"Enhancement Error: {str(e)}"
    
    def _make_request(self, provider: LLMProvider, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Make a request to the specified provider."""
        model = self.models[provider]
        
        if provider == LLMProvider.ANTHROPIC:
            return self._make_anthropic_request(prompt, max_tokens, temperature)
        elif provider == LLMProvider.OPENROUTER:
            return self._make_openrouter_request(prompt, max_tokens, temperature)
        elif provider == LLMProvider.LOCAL:
            return self._make_local_request(prompt, max_tokens, temperature)
        
        return None
    
    def _make_anthropic_request(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Make request to Anthropic API."""
        if not ANTHROPIC_AVAILABLE or not self.config.secrets.get("anthropic_api_key"):
            return None
        
        try:
            client = anthropic.Anthropic(api_key=self.config.secrets["anthropic_api_key"])
            model = self.models[LLMProvider.ANTHROPIC]
            
            response = client.messages.create(
                model=model.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"Anthropic request failed: {e}")
            return None
    
    def _make_openrouter_request(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Make request to OpenRouter API."""
        if not self.config.secrets.get("openrouter_api_key"):
            return None
        
        try:
            model = self.models[LLMProvider.OPENROUTER]
            payload = {
                "model": model.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **self.config.provider_routing
            }
            
            response = requests.post(
                model.endpoint,
                headers=model.headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"OpenRouter request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"OpenRouter request failed: {e}")
            return None
    
    def _make_local_request(self, prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
        """Make request to local LM Studio."""
        try:
            model = self.models[LLMProvider.LOCAL]
            payload = {
                "model": model.model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            response = requests.post(
                model.endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Local request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Local request failed: {e}")
            return None

# Initialize global configuration and client
CONFIG = AppConfig()
LLM_CLIENT = LLMClient(CONFIG)

# Utility functions
def ensure_directories():
    """Create temporary and journal directories if they don't exist."""
    os.makedirs(CONFIG.temp_dir, exist_ok=True)
    os.makedirs(CONFIG.journal_dir, exist_ok=True)

def get_file_metadata(file_path: str) -> datetime:
    """Get file creation and modification times."""
    try:
        stat = os.stat(file_path)
        return datetime.fromtimestamp(stat.st_mtime)
    except:
        return datetime.now()

def get_video_recording_time(file_path: str) -> datetime:
    """Extract recording time from video metadata using ffprobe."""
    try:
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
            creation_time_str = result.stdout.strip().split('\\n')[0]
            if creation_time_str and creation_time_str != "N/A":
                for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        dt = datetime.strptime(creation_time_str, fmt)
                        if creation_time_str.endswith('Z'):
                            dt = dt.replace(tzinfo=timezone.utc).astimezone()
                            dt = dt.replace(tzinfo=None)
                        return dt
                    except ValueError:
                        continue
        
        print(f"Using file modification time for {Path(file_path).name}")
        return get_file_metadata(file_path)
        
    except Exception as e:
        print(f"Error getting video metadata for {Path(file_path).name}: {e}")
        return get_file_metadata(file_path)

def parse_file_paths(input_text: str) -> List[str]:
    """Parse file paths from input text."""
    if not input_text or not input_text.strip():
        return []
    
    lines = [line.strip() for line in input_text.strip().split('\\n') if line.strip()]
    
    if len(lines) == 1:
        single_line = lines[0]
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v', '.webm', '.flv', '.wmv']
        extension_count = sum(single_line.lower().count(ext) for ext in video_extensions)
        
        if extension_count > 1:
            parts = single_line.split()
            paths = []
            current_path = ""
            
            for part in parts:
                if current_path:
                    current_path += " " + part
                else:
                    current_path = part
                
                if any(current_path.lower().endswith(ext) for ext in video_extensions):
                    paths.append(current_path)
                    current_path = ""
            
            if current_path:
                paths.append(current_path)
            
            return paths
    
    return lines

def convert_to_mp3(video_path: str, progress_callback=None) -> str:
    """Convert video file to MP3 format."""
    video_name = Path(video_path).stem
    mp3_path = os.path.join(CONFIG.temp_dir, f"{video_name}.mp3")

    if progress_callback:
        progress_callback("Converting to MP3...")

    cmd = [CONFIG.ffmpeg_path, "-y", "-i", video_path, "-q:a", "0", "-map", "a", mp3_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return mp3_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error converting video to MP3: {e.stderr}")

def transcribe_audio(mp3_path: str, progress_callback=None) -> str:
    """Transcribe MP3 file using parakeet-mlx."""
    if progress_callback:
        progress_callback("Transcribing audio...")
        
    try:
        cmd = [
            "parakeet-mlx",
            "--output-format", "txt",
            "--output-dir", CONFIG.temp_dir,
            mp3_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        txt_path = os.path.join(CONFIG.temp_dir, f"{Path(mp3_path).stem}.txt")

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                transcript = f.read()
            os.remove(txt_path)
            return transcript
        else:
            raise Exception("Transcription file was not created")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error during transcription: {e.stderr}")

def create_text_diff(original: str, enhanced: str) -> str:
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

def create_journal_entry(video_name: str, file_timestamp: datetime, transcript: str, brief_title: str = "") -> str:
    """Create a properly formatted header-based journal entry."""
    date_str = file_timestamp.strftime("%Y-%m-%d")
    time_str = file_timestamp.strftime("%H:%M:%S")
    
    display_title = brief_title if brief_title else video_name
    
    return f"""## {display_title} ({date_str} {time_str})
### Video: {video_name}

{transcript}

---

"""

def append_to_journal(journal_entry: str, file_timestamp: datetime, custom_path: Optional[str] = None, progress_callback=None) -> str:
    """Append the journal entry to the appropriate daily journal file."""
    if progress_callback:
        progress_callback("Saving to journal...")
        
    if custom_path:
        journal_path = custom_path
    else:
        date_str = file_timestamp.strftime("%Y-%m-%d")
        journal_path = os.path.join(CONFIG.journal_dir, f"{date_str}-auto.md")

    if not os.path.exists(journal_path):
        header = f"# Daily Auto Journal - {file_timestamp.strftime('%B %d, %Y')}\\n\\n"
        with open(journal_path, "w") as f:
            f.write(header)

    with open(journal_path, "a") as f:
        f.write(journal_entry)

    return journal_path

# Main processing functions
def process_single_video(
    video_file,
    auto_save: bool,
    custom_journal_path: str,
    enhance_with_llm: bool,
    enhancement_prompt: str,
    provider: str,
    progress_callback=None,
) -> Tuple[str, str, str, Optional[str]]:
    """Process a single video file and return formatted transcript."""
    video_path = video_file.name
    video_name = Path(video_path).stem
    
    # Get recording time
    if hasattr(video_file, 'recording_time'):
        file_timestamp = video_file.recording_time
    else:
        file_timestamp = get_video_recording_time(video_path)
    
    print(f"\\n=== Processing {video_name} ===")
    print(f"Provider: {provider}")
    print(f"Enhance with LLM: {enhance_with_llm}")

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
                
                llm_provider = LLMProvider(provider)
                
                # Generate brief title
                if progress_callback:
                    progress_callback("Generating title...")
                brief_title = LLM_CLIENT.generate_title(original_transcript, llm_provider)
                
                # Enhance transcript
                if progress_callback:
                    progress_callback("Enhancing transcript...")
                enhanced_transcript, error = LLM_CLIENT.enhance_transcript(
                    original_transcript, enhancement_prompt, llm_provider
                )

                if enhanced_transcript and enhanced_transcript != original_transcript:
                    final_transcript = enhanced_transcript
                    diff_output = create_text_diff(original_transcript, enhanced_transcript)
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
            else:
                enhancement_status = "⏭️ LLM enhancement disabled"

            # Create journal entry
            journal_entry = create_journal_entry(video_name, file_timestamp, final_transcript, brief_title)

            # Auto-save to journal if enabled
            journal_path = None
            if auto_save:
                journal_path = append_to_journal(
                    journal_entry,
                    file_timestamp,
                    custom_journal_path if custom_journal_path.strip() else None,
                    progress_callback,
                )

            # Format display transcript
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
    file_paths_input: str,
    video_files,
    auto_save: bool,
    custom_journal_path: str,
    enhance_with_llm: bool,
    enhancement_prompt: str,
    provider: str,
    progress=gr.Progress(),
) -> Tuple[str, str, str]:
    """Main processing function for multiple video files."""
    
    print(f"\\n=== Starting batch processing ===")
    print(f"Provider: {provider}")
    print(f"LLM Enhancement enabled: {enhance_with_llm}")

    # Prepare file list
    files_to_process = []

    if file_paths_input and file_paths_input.strip():
        progress(0.0, desc="Parsing file paths...")
        file_paths = parse_file_paths(file_paths_input)
        
        # Validate files exist
        missing_files = []
        valid_files = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        if missing_files:
            return f"Files not found:\\n" + "\\n".join(f"- {path}" for path in missing_files), "", ""
        
        # Sort by recording time
        progress(0.02, desc="Reading video metadata...")
        files_with_times = []
        
        for file_path in valid_files:
            recording_time = get_video_recording_time(file_path)
            files_with_times.append((file_path, recording_time))
        
        files_with_times.sort(key=lambda x: x[1])
        
        # Create mock file objects
        class MockFile:
            def __init__(self, path, recording_time):
                self.name = path
                self.recording_time = recording_time

        files_to_process = [MockFile(path, time) for path, time in files_with_times]

    elif video_files and len(video_files) > 0:
        progress(0.0, desc="Preparing uploaded files...")
        
        # Sort uploaded files by recording time
        files_with_times = []
        for video_file in video_files:
            recording_time = get_video_recording_time(video_file.name)
            files_with_times.append((video_file, recording_time))
        
        files_with_times.sort(key=lambda x: x[1])
        files_to_process = [video_file for video_file, _ in files_with_times]

    if not files_to_process:
        return "Please provide video files via file paths or upload.", "", ""

    ensure_directories()

    # Process files
    all_transcripts = []
    all_journal_entries = []
    all_diffs = []
    total_files = len(files_to_process)

    # Add summary header
    summary = f"""# Video Transcription Results
**Total Files:** {total_files}
**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Provider:** {provider}
**Auto-save:** {'Enabled' if auto_save else 'Disabled'}
**LLM Enhancement:** {'Enabled' if enhance_with_llm else 'Disabled'}

"""
    all_transcripts.append(summary)

    # Process each file
    for i, video_file in enumerate(files_to_process):
        progress_start = 0.1 + (i * 0.8 / total_files)
        progress_end = 0.1 + ((i + 1) * 0.8 / total_files)
        
        video_name = Path(video_file.name).stem
        
        def file_progress_callback(step_description):
            current_progress = progress_start + (progress_end - progress_start) * 0.5
            progress(current_progress, desc=f"[{i+1}/{total_files}] {video_name}: {step_description}")

        progress(progress_start, desc=f"[{i+1}/{total_files}] Starting {video_name}")

        transcript, journal_entry, diff_output, error = process_single_video(
            video_file,
            auto_save,
            custom_journal_path,
            enhance_with_llm,
            enhancement_prompt,
            provider,
            file_progress_callback,
        )

        all_transcripts.append(transcript)
        if journal_entry:
            all_journal_entries.append(journal_entry)
        if diff_output:
            all_diffs.append(f"## {Path(video_file.name).stem}\\n\\n```diff\\n{diff_output}\\n```\\n\\n")

        progress(progress_end, desc=f"[{i+1}/{total_files}] Completed {video_name}")

    progress(1.0, desc="All files processed successfully!")
    return "".join(all_transcripts), "".join(all_journal_entries), "".join(all_diffs)

def create_interface():
    """Create and configure the Gradio interface."""
    
    default_prompt = """You are an expert transcript editor. Your task is to enhance a raw transcript while preserving authenticity.

## Pre-Processing Check
First, determine if the transcript should be processed:
- If less than 50 meaningful words, return only: SKIP: Too short
- If mostly silence/non-verbal, return only: SKIP: No meaningful content
- If corrupted/incoherent, return only: SKIP: Corrupted transcript

## Enhancement Focus Areas
1. **Remove filler words:** um, uh, like, you know (unless meaningful)
2. **Fix transcription errors:** obvious word mistakes, repeated words
3. **Add structure:**
   - Insert paragraph breaks between distinct thoughts
   - Add topic headers for new subjects: ### *Topic: [Name]*
   - Add context notes for clarity: ***[context: description]***
4. **Preserve everything else:** Keep original words, repetitions, tangents, speaker voice

## Rules
- Be conservative with changes
- Preserve meaning and authenticity
- Only suggest necessary improvements
- When in doubt, preserve the original"""

    with gr.Blocks(theme=gr.themes.Base(), title="Video Transcription Tool") as interface:
        gr.Markdown("# Video Transcription Tool")
        gr.Markdown("Upload video files to generate transcripts with optional LLM enhancement and automatic journal integration.")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("File Paths"):
                        file_paths_input = gr.Textbox(
                            label="Video File Paths",
                            lines=8,
                            placeholder="Paste file paths here:\\n\\nOne per line:\\n/path/to/video1.mp4\\n/path/to/video2.mp4",
                            info="Enter full paths to your video files",
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
                choices=["anthropic", "openrouter", "local"],
                value="anthropic",
                label="LLM Provider",
                info="Choose your preferred LLM provider"
            )

            # Provider-specific settings
            with gr.Group() as anthropic_settings:
                gr.Markdown("### Anthropic API Settings")
                gr.Markdown(f"**Model:** {LLM_CLIENT.models[LLMProvider.ANTHROPIC].model_id}")
                if CONFIG.secrets.get("anthropic_api_key"):
                    gr.Markdown("✅ **API Key:** Configured")
                else:
                    gr.Markdown("❌ **API Key:** Not set")

            with gr.Group(visible=False) as openrouter_settings:
                gr.Markdown("### OpenRouter API Settings")
                gr.Markdown(f"**Model:** {LLM_CLIENT.models[LLMProvider.OPENROUTER].model_id}")
                gr.Markdown(f"**Provider Routing:** FP8+ precision, no data collection")
                if CONFIG.secrets.get("openrouter_api_key"):
                    gr.Markdown("✅ **API Key:** Configured")
                else:
                    gr.Markdown("❌ **API Key:** Not set")

            with gr.Group(visible=False) as local_settings:
                gr.Markdown("### Local LM Studio Settings")
                gr.Markdown(f"**Endpoint:** {CONFIG.local_endpoint}")
                gr.Markdown(f"**Model:** {CONFIG.local_model}")

            # Test connection button
            test_model_btn = gr.Button("🧪 Test Model Connection", variant="secondary", size="sm")
            test_output = gr.Markdown(label="Test Results", value="", visible=False)

            enhancement_prompt = gr.Textbox(
                value=default_prompt,
                label="Enhancement Prompt",
                lines=6,
                info="Instructions for the LLM on how to improve the transcript",
            )

        # Process button
        process_btn = gr.Button("Process Videos", variant="primary", size="lg")

        # Output tabs
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
                    label="Journal Entries",
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
                )

        # Event handlers
        def update_provider_settings(provider_choice):
            return {
                anthropic_settings: gr.Group(visible=provider_choice == "anthropic"),
                openrouter_settings: gr.Group(visible=provider_choice == "openrouter"),
                local_settings: gr.Group(visible=provider_choice == "local")
            }

        def test_model_connection(provider_choice):
            try:
                provider_enum = LLMProvider(provider_choice)
                success, message = LLM_CLIENT.test_connection(provider_enum)
                return {test_output: gr.Markdown(value=message, visible=True)}
            except Exception as e:
                return {test_output: gr.Markdown(value=f"❌ **Error:** {str(e)}", visible=True)}

        provider.change(
            fn=update_provider_settings,
            inputs=[provider],
            outputs=[anthropic_settings, openrouter_settings, local_settings]
        )

        test_model_btn.click(
            fn=test_model_connection,
            inputs=[provider],
            outputs=[test_output]
        )

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
            ],
            outputs=[transcript_output, journal_output, diff_output],
        )

        gr.Markdown("""
        ### Features:
        - **Unified LLM Interface**: Single codebase supporting multiple providers
        - **OpenRouter Provider Routing**: FP8+ precision, no data collection
        - **Centralized Configuration**: Easy to modify and extend
        - **Enhanced Error Handling**: Robust error handling across all providers
        - **Automatic Fallbacks**: Smart fallbacks for token limits and failures
        
        ### Provider Setup:
        - **Anthropic**: Add API key to secrets.json
        - **OpenRouter**: Add API key to secrets.json (uses privacy-focused routing)
        - **Local**: Ensure LM Studio is running
        """)

    return interface

def main():
    """Launch the Gradio app."""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0", 
        server_port=7868, 
        share=False, 
        show_error=True
    )

if __name__ == "__main__":
    main()