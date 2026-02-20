# 🎙️ PodCasteer - AI Camera Switcher with Whisper

Smart camera switching app that uses **OpenAI Whisper** for real-time speech recognition and speaker diarization to automatically switch OBS camera sources based on who is speaking.

## 🚀 What's New in Whisper Edition

- **🎯 Real-time Speech Recognition** - Powered by OpenAI Whisper
- **👥 Speaker Diarization** - Identifies who is speaking and switches to their camera
- **🤖 AI Smart Modes**:
  - **Speaker Focus** - Automatically follows the active speaker
  - **Voice Activity** - Dynamic switching based on conversation flow
  - **Reaction Detection** - Switches to reaction cam on excitement keywords
- **📝 Live Transcription** - See what everyone is saying in real-time
- **🎚️ Easy Calibration** - Record voice samples to train speaker recognition

## 📋 Requirements

- **OBS Studio** with WebSocket plugin enabled
- **Python 3.9+** for the Whisper server
- **Modern browser** (Chrome/Firefox/Edge)
- **Microphone** for speech input

## 🛠️ Installation

### 1. Start the Whisper Server

```bash
cd server
pip install -r requirements.txt
python whisper_server.py
```

The server will:
- Download Whisper model (~150MB for base model)
- Start on `ws://localhost:8765`

### 2. Open the Web App

Simply open `index.html` in your browser:

```bash
# Option 1: Double-click index.html
# Option 2: Use a local server
npx serve .
```

## 🎮 Quick Start

### 1. Connect to Server

1. Click **"Connect to Server"** (default: `ws://localhost:8765`)
2. Wait for Whisper to load (first time takes ~30 seconds)

### 2. Connect to OBS

1. In OBS: **Tools → WebSocket Server Settings**
2. Enable WebSocket server (default port: 4455)
3. In PodCasteer: Click **"Connect OBS"**

### 3. Configure Cameras

1. Enter your OBS scene name
2. Enter camera source names (comma-separated)
3. These should match exactly as they appear in OBS

### 4. Set Up Speakers

1. Click **"Add Speaker"** for each person
2. Assign them a camera
3. Click **"🎤 Calibrate"** and speak for 5 seconds
4. Repeat for all speakers

### 5. Start Whisper

1. Click **"Start Whisper"**
2. Allow microphone access
3. You'll see live transcription appear

### 6. Start AI Switcher

1. Select **AI Smart** mode
2. Choose AI mode (Speaker Focus recommended)
3. Click **"Start AI Switcher"**
4. Start streaming!

## 🎯 AI Modes Explained

### Speaker Focus (Recommended)
The AI identifies who is speaking and switches to their assigned camera automatically. Perfect for:
- Podcasts with multiple hosts
- Panel discussions
- Interview formats

### Voice Activity
Switches cameras dynamically based on speech patterns. Good for:
- Fast-paced conversations
- Shows where anyone might speak
- Variety and engagement

### Reaction Detection
Stays on main camera normally, but switches to reaction camera when someone says exciting things like:
- "Wow!", "OMG!", "No way!"
- Detects exclamations and excitement
- Automatically returns after 3 seconds

## 📐 Camera Setup in OBS

```
Scene
├── [✓] Camera 1    (Zekki - Main)
├── [ ] Camera 2    (Guest 1)
├── [ ] Camera 3    (Guest 2)
└── [ ] Reaction Cam (Close-up)
```

In PodCasteer, enter: `Camera 1, Camera 2, Camera 3, Reaction Cam`

Register speakers and assign cameras:
- Zekki → Camera 1
- Guest 1 → Camera 2
- Guest 2 → Camera 3

## 🔧 Advanced Configuration

### Whisper Model Size

Edit `whisper_server.py` to change the model:

```python
# Line ~320
await self.initialize_whisper(model_size="small")  # More accurate, slower
```

Options: `tiny` (fastest), `base` (balanced), `small` (accurate), `medium` (very accurate)

### Switch Cooldown

Set minimum time between camera switches to avoid rapid switching:
- Default: 2 seconds
- Adjust in the AI Control panel

### Calibration Tips

- Record in a quiet environment
- Speak naturally for 5 seconds
- Each speaker should calibrate separately
- Re-calibrate if voice recognition seems off

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Failed to connect to server" | Make sure `python whisper_server.py` is running |
| Whisper takes forever to load | First startup downloads the model (~150MB) |
| Speaker not recognized | Re-calibrate with a longer, clearer sample |
| Cameras not switching | Check OBS scene/item names match exactly |
| Transcription is wrong | Try `small` model for better accuracy |
| High CPU usage | Use `tiny` model or reduce audio buffer size |

## 🎨 Browser Compatibility

- ✅ Chrome/Edge: Full support
- ✅ Firefox: Full support
- ⚠️ Safari: May need microphone permissions

## 🔄 Architecture

```
┌─────────────┐     WebSocket      ┌─────────────────┐
│   Browser   │ ←───────────────→ │  Whisper Server │
│  (Frontend) │    (Audio + Text) │   (Python)      │
└──────┬──────┘                   └────────┬────────┘
       │                                    │
       │ WebRTC Audio                       │ Whisper
       │                                    │ Transcription
       ↓                                    ↓
┌─────────────┐                   ┌─────────────────┐
│  Microphone │                   │  Speaker ID     │
│   Input     │                   │  + Diarization  │
└─────────────┘                   └────────┬────────┘
                                          │
                                          ↓
                                   ┌─────────────────┐
                                   │   OBS Switch    │
                                   │   WebSocket     │
                                   └─────────────────┘
```

## 🚧 Future Enhancements

- [ ] Multiple microphone support (for co-hosts)
- [ ] Face detection integration
- [ ] Custom switching rules via Lua/JS
- [ ] StreamDeck integration
- [ ] Standalone desktop app (Electron/Tauri)
- [ ] Cloud Whisper API option (no local GPU needed)
- [ ] Sentiment analysis for smarter reactions
- [ ] Custom keyword triggers

---

Built for streamers by streamers 🎮

**Zekki** - Now with AI! 🤖
