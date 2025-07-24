# API Key Management Guidelines

## Overview
This document provides guidance on securely managing API keys across all projects in this workspace.

## Current Implementation
All API keys should be stored in `secrets.json` files within each project directory, following this pattern:

### secrets.json Structure
```json
{
  "anthropic_api_key": "sk-ant-api03-...",
  "openrouter_api_key": "sk-or-v1-...",
  "comment": "Add your API keys here. This file should be kept private and not shared."
}
```

## Loading Pattern
Projects should implement a secrets loading function like this:

```python
def load_secrets():
    """Load API keys from secrets.json file."""
    secrets_path = os.path.join(os.path.dirname(__file__), "secrets.json")
    try:
        with open(secrets_path, 'r') as f:
            secrets = json.load(f)
        return secrets
    except FileNotFoundError:
        print(f"Warning: secrets.json not found at {secrets_path}. Using environment variables as fallback.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing secrets.json: {e}. Using environment variables as fallback.")
        return {}

# Load secrets at startup
SECRETS = load_secrets()

# Use with fallback to environment variables
API_KEY = SECRETS.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY", ""))
```

## Security Best Practices

###  DO:
- Store all API keys in `secrets.json` files
- Add `secrets.json` to `.gitignore` immediately
- Provide `secrets_template.json` with empty/example values for setup
- Use environment variables as fallbacks
- Show clear status messages about key loading source
- Include error handling for missing/malformed secrets files

### L DON'T:
- Hardcode API keys directly in source code
- Commit actual API keys to version control
- Store keys in configuration files that might be shared
- Log or print actual API key values

## Implementation Status

### AudioTools/GradioTranscribeToJournal.py 
- **Status**: Updated to use secrets.json
- **Keys**: `anthropic_api_key`, `openrouter_api_key`  
- **Fallback**: Environment variables
- **Location**: `/Users/mini/Documents/AudioTools/secrets.json`

### Future Projects
When creating new projects that require API keys:

1. Create `secrets.json` with required keys
2. Create `secrets_template.json` with placeholder values
3. Add `secrets.json` to `.gitignore`
4. Implement the loading pattern above
5. Update this document with implementation status

## Testing
Always verify API key loading works correctly:
- Test with secrets.json present
- Test with secrets.json missing (fallback to env vars)
- Test with malformed JSON (error handling)
- Use the "Test Model Connection" features where available

## Emergency Key Rotation
If API keys are compromised:
1. Immediately revoke old keys at the provider
2. Generate new keys
3. Update all `secrets.json` files
4. Update any environment variables
5. Test all affected applications

---
*Last updated: 2025-07-24*