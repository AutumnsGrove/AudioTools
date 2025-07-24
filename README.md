# AudioTools - Video Transcription Tool

A comprehensive video transcription tool with multiple LLM provider support and automatic journal integration.

## Features

- **Video Transcription**: Converts videos to text using parakeet-mlx
- **Triple LLM Support**: Choose between local LM Studio, Anthropic Claude Sonnet, or OpenRouter (Kimi K2)
- **Instruction-Based Diff Enhancement**: LLM provides specific edit instructions instead of full rewrites
- **Model Testing**: Test your LLM connection before processing videos
- **Brief Titles**: Auto-generated descriptive titles (8 words max) for each transcript
- **Auto-Journal**: Automatically saves enhanced transcripts to daily journals
- **Batch Processing**: Handle multiple videos at once
- **Chronological Processing**: Automatically sorts videos by recording time

## Setup

### 1. Install Dependencies

```bash
pip install gradio anthropic requests
```

### 2. Configure API Keys

Create a `secrets.json` file in the same directory as the script:

```json
{
  "anthropic_api_key": "your-anthropic-api-key-here",
  "openrouter_api_key": "your-openrouter-api-key-here",
  "comment": "Add your API keys here. This file should be kept private and not shared."
}
```

### 3. API Key Setup

#### Anthropic API
1. Sign up at [Anthropic](https://console.anthropic.com/)
2. Create an API key
3. Add it to `secrets.json` as `anthropic_api_key`

#### OpenRouter API (for Kimi K2)
1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Create an API key
3. Add it to `secrets.json` as `openrouter_api_key`

#### Local LM Studio
1. Install [LM Studio](https://lmstudio.ai/)
2. Load a model
3. Start the local server (usually http://localhost:1234)

### 4. Run the Application

```bash
python GradioTranscribeToJournal.py
```

The web interface will open at `http://localhost:7868`

## Usage

1. **Choose LLM Provider**: Select between local, anthropic, or openrouter
2. **Test Connection**: Use the "Test Model Connection" button to verify setup
3. **Upload Videos**: Either drag & drop files or paste file paths
4. **Process**: Click "Process Videos" to transcribe and enhance

## Models

- **Anthropic**: claude-sonnet-4-20250514
- **OpenRouter**: moonshot-v1-8k (Kimi K2)
- **Local**: Your choice of model in LM Studio

## Security

- API keys are stored in `secrets.json` (not committed to git)
- The `.gitignore` file prevents accidental commits of secrets
- Never share your `secrets.json` file

## File Structure

```
AudioTools/
├── GradioTranscribeToJournal.py  # Main application
├── secrets.json                  # API keys (create this)
├── .gitignore                    # Prevents committing secrets
└── README.md                     # This file
```