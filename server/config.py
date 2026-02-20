"""
PodCasteer Server Configuration
"""

# Whisper Model Settings
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
WHISPER_DEVICE = "auto"  # auto, cpu, cuda

# Server Settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765

# Audio Settings
SAMPLE_RATE = 16000
BUFFER_SECONDS = 3

# AI Switching Settings
DEFAULT_COOLDOWN = 2.0  # Minimum seconds between switches
DEFAULT_MODE = "speaker-focus"  # speaker-focus, voice-activity, reaction-cam

# OBS Settings
OBS_DEFAULT_HOST = "localhost"
OBS_DEFAULT_PORT = 4455
