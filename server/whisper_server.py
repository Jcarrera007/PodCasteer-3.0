#!/usr/bin/env python3
"""
PodCasteer Whisper Server
Real-time speech recognition with speaker diarization for AI camera switching
"""

import asyncio
import json
import wave
import io
import os
import tempfile
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# Whisper imports
from faster_whisper import WhisperModel

# OBS WebSocket
from obswebsocket import obsws, requests as obs_requests


class SimpleSpeakerDiarizer:
    """Lightweight speaker diarization using audio features"""
    
    def __init__(self):
        # Speaker database: {speaker_id: embedding}
        self.known_speakers: Dict[str, np.ndarray] = {}
        self.speaker_names: Dict[str, str] = {}
        self.speaker_counter = 0
        
        # Camera mapping: {speaker_id: camera_index}
        self.speaker_camera_map: Dict[str, int] = {}
        
        # Recent speaker history
        self.speaker_history = deque(maxlen=5)
        self.current_speaker = "unknown"
        
    def add_speaker(self, name: str, camera_index: int) -> str:
        """Register a new speaker with a name and camera"""
        speaker_id = f"speaker_{self.speaker_counter}"
        self.speaker_counter += 1
        self.speaker_names[speaker_id] = name
        self.speaker_camera_map[speaker_id] = camera_index
        print(f"Registered speaker: {name} (ID: {speaker_id}) -> Camera {camera_index}")
        return speaker_id
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract simple audio features for speaker identification"""
        # Normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Basic features
        features = []
        
        # MFCC-like features (simplified)
        # Use FFT-based spectral features
        fft = np.abs(np.fft.rfft(audio_data))
        
        # Spectral centroid (brightness)
        freqs = np.fft.rfftfreq(len(audio_data), 1/16000)
        spectral_centroid = np.sum(freqs * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
        features.append(spectral_centroid / 8000)  # Normalize
        
        # Spectral rolloff
        cumsum = np.cumsum(fft)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        features.append(rolloff / 8000)
        
        # Zero crossing rate (noisiness)
        zcr = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        features.append(zcr)
        
        # RMS energy
        rms = np.sqrt(np.mean(audio_data**2))
        features.append(rms)
        
        # Pitch estimation (simplified autocorrelation)
        correlation = np.correlate(audio_data, audio_data, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Find first peak for pitch
        if len(correlation) > 100:
            peak_idx = np.argmax(correlation[50:400]) + 50
            pitch = 16000 / peak_idx if peak_idx > 0 else 0
            features.append(pitch / 500)  # Normalize
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def identify_speaker(self, audio_data: np.ndarray) -> Tuple[str, float]:
        """Identify speaker from audio features"""
        features = self.extract_features(audio_data)
        
        if not self.known_speakers:
            return "unknown", 0.0
        
        best_match = None
        best_score = -1
        
        for speaker_id, known_features in self.known_speakers.items():
            # Euclidean distance (inverted for similarity)
            distance = np.linalg.norm(features - known_features)
            similarity = 1 / (1 + distance)
            
            if similarity > best_score:
                best_score = similarity
                best_match = speaker_id
        
        # Threshold for new speaker
        if best_score < 0.6:
            return "unknown", best_score
        
        return best_match, best_score
    
    def register_speaker_sample(self, speaker_id: str, audio_data: np.ndarray):
        """Register a voice sample for a speaker"""
        features = self.extract_features(audio_data)
        
        if speaker_id in self.known_speakers:
            # Average with existing features for refinement
            self.known_speakers[speaker_id] = (
                self.known_speakers[speaker_id] * 0.7 + features * 0.3
            )
        else:
            self.known_speakers[speaker_id] = features
        
        print(f"Calibrated speaker {speaker_id} ({self.speaker_names.get(speaker_id, 'Unknown')})")
    
    def update_speaker_history(self, speaker_id: str):
        """Update speaker history and get dominant speaker"""
        self.speaker_history.append(speaker_id)
        
        if not self.speaker_history:
            return "unknown"
        
        # Count occurrences
        speaker_counts = {}
        for s in self.speaker_history:
            speaker_counts[s] = speaker_counts.get(s, 0) + 1
        
        dominant = max(speaker_counts, key=speaker_counts.get)
        self.current_speaker = dominant
        return dominant
    
    def get_current_speaker_camera(self) -> int:
        """Get camera index for current speaker"""
        return self.speaker_camera_map.get(self.current_speaker, 0)


class WhisperProcessor:
    """Handles Whisper transcription with speaker diarization"""
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        print(f"Loading Whisper model: {model_size} on {device}")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        
        # Initialize lightweight speaker diarizer
        self.diarizer = SimpleSpeakerDiarizer()
        
    def add_speaker(self, name: str, camera_index: int) -> str:
        """Register a new speaker"""
        return self.diarizer.add_speaker(name, camera_index)
    
    def register_speaker_sample(self, speaker_id: str, audio_data: np.ndarray):
        """Calibrate speaker with audio sample"""
        self.diarizer.register_speaker_sample(speaker_id, audio_data)
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[dict]:
        """Transcribe audio and identify speakers"""
        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            with wave.open(f, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data.tobytes())
        
        try:
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                temp_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            results = []
            for segment in segments:
                # Extract segment audio for speaker ID
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                
                if end_sample > start_sample and end_sample <= len(audio_data):
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    # Identify speaker
                    speaker_id, confidence = self.diarizer.identify_speaker(segment_audio)
                    
                    # Update history
                    dominant_speaker = self.diarizer.update_speaker_history(speaker_id)
                    
                    results.append({
                        "text": segment.text.strip(),
                        "speaker_id": speaker_id,
                        "speaker_name": self.diarizer.speaker_names.get(speaker_id, "Unknown"),
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": confidence,
                        "dominant_speaker": dominant_speaker,
                        "camera": self.diarizer.get_current_speaker_camera()
                    })
            
            return results
            
        finally:
            os.unlink(temp_path)


class OBSSwitcher:
    """Controls OBS camera switching"""
    
    def __init__(self):
        self.ws = None
        self.scene_name = "Scene"
        self.camera_sources: List[str] = []
        self.connected = False
    
    def connect(self, host: str = "localhost", port: int = 4455, password: str = ""):
        """Connect to OBS WebSocket"""
        try:
            self.ws = obsws(host, port, password)
            self.ws.connect()
            self.connected = True
            print(f"Connected to OBS at {host}:{port}")
            return True
        except Exception as e:
            print(f"Failed to connect to OBS: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from OBS"""
        if self.ws:
            self.ws.disconnect()
            self.connected = False
    
    def set_scene_name(self, name: str):
        """Set the active scene name"""
        self.scene_name = name
    
    def set_camera_sources(self, sources: List[str]):
        """Set the list of camera source names"""
        self.camera_sources = sources
    
    def switch_to_camera(self, index: int) -> bool:
        """Switch to camera by index"""
        if not self.connected or index < 0 or index >= len(self.camera_sources):
            return False
        
        try:
            target_source = self.camera_sources[index]
            
            # Get scene items
            scene_items = self.ws.call(obs_requests.GetSceneItemList(sceneName=self.scene_name))
            
            # Enable target camera, disable others
            for item in scene_items.getSceneItems():
                source_name = item["sourceName"]
                should_be_visible = source_name == target_source
                
                self.ws.call(obs_requests.SetSceneItemEnabled(
                    sceneName=self.scene_name,
                    sceneItemId=item["sceneItemId"],
                    sceneItemEnabled=should_be_visible
                ))
            
            print(f"Switched to camera: {target_source}")
            return True
            
        except Exception as e:
            print(f"Error switching camera: {e}")
            return False


class PodCasteerServer:
    """Main WebSocket server handling audio streaming and AI switching"""
    
    def __init__(self):
        self.whisper = None
        self.obs = OBSSwitcher()
        self.clients: set[WebSocketServerProtocol] = set()
        
        # Audio buffer for streaming (3 seconds at 16kHz, 16-bit)
        self.audio_buffer = bytearray()
        self.buffer_size = 16000 * 2 * 3
        self.processing = False
        
        # Switching state
        self.current_speaker = None
        self.last_switch_time = 0
        self.switch_cooldown = 2.0
        
        # Mode settings
        self.mode = "manual"  # manual, auto, ai
        self.ai_mode = "speaker-focus"  # speaker-focus, voice-activity, reaction-cam
        
    async def initialize_whisper(self, model_size: str = "base"):
        """Initialize Whisper model"""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.whisper = WhisperProcessor(model_size=model_size, device=device)
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket client connections"""
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected: {websocket.remote_address}")
    
    async def handle_message(self, websocket: WebSocketServerProtocol, message):
        """Process incoming messages"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "audio_chunk":
                # Receive audio data
                audio_bytes = bytes(data["data"])
                await self.process_audio_chunk(audio_bytes)
                
            elif msg_type == "obs_connect":
                success = self.obs.connect(
                    data.get("host", "localhost"),
                    data.get("port", 4455),
                    data.get("password", "")
                )
                await self.send_to_client(websocket, {
                    "type": "obs_status",
                    "connected": success
                })
                
            elif msg_type == "register_speaker":
                speaker_id = self.whisper.add_speaker(
                    data["name"],
                    data["camera_index"]
                )
                await self.send_to_client(websocket, {
                    "type": "speaker_registered",
                    "speaker_id": speaker_id,
                    "name": data["name"],
                    "camera": data["camera_index"]
                })
                
            elif msg_type == "calibrate_speaker":
                await self.calibrate_speaker(data)
                await self.send_to_client(websocket, {
                    "type": "speaker_calibrated",
                    "speaker_id": data["speaker_id"]
                })
                
            elif msg_type == "switch_camera":
                self.obs.switch_to_camera(data["camera_index"])
                
            elif msg_type == "set_mode":
                self.mode = data["mode"]
                self.ai_mode = data.get("ai_mode", "speaker-focus")
                if "cooldown" in data:
                    self.switch_cooldown = data["cooldown"]
                await self.broadcast({
                    "type": "mode_changed",
                    "mode": self.mode,
                    "ai_mode": self.ai_mode
                })
                
            elif msg_type == "config":
                if "scene_name" in data:
                    self.obs.set_scene_name(data["scene_name"])
                if "camera_sources" in data:
                    self.obs.set_camera_sources(data["camera_sources"])
                    
        except Exception as e:
            print(f"Error handling message: {e}")
            await self.send_to_client(websocket, {
                "type": "error",
                "message": str(e)
            })
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Buffer audio and process when ready"""
        self.audio_buffer.extend(audio_data)
        
        if len(self.audio_buffer) >= self.buffer_size and not self.processing:
            self.processing = True
            
            # Extract buffer
            process_bytes = bytes(self.audio_buffer[:self.buffer_size])
            self.audio_buffer = self.audio_buffer[self.buffer_size // 2:]  # Keep half
            
            # Convert to numpy
            audio_np = np.frombuffer(process_bytes, dtype=np.int16)
            
            # Process in background
            asyncio.create_task(self.process_audio_segment(audio_np))
    
    async def process_audio_segment(self, audio_np: np.ndarray):
        """Process audio with Whisper and handle AI switching"""
        try:
            results = self.whisper.transcribe_audio(audio_np)
            
            if results:
                latest = results[-1]
                
                # Broadcast transcription
                await self.broadcast({
                    "type": "transcription",
                    "text": latest["text"],
                    "speaker": latest["speaker_name"],
                    "speaker_id": latest["speaker_id"],
                    "camera": latest["camera"],
                    "confidence": latest["confidence"]
                })
                
                # Handle AI switching
                if self.mode == "ai":
                    await self.handle_ai_switching(latest)
            
            self.processing = False
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            self.processing = False
    
    async def handle_ai_switching(self, segment: dict):
        """Handle AI-based camera switching"""
        now = time.time()
        
        if now - self.last_switch_time < self.switch_cooldown:
            return
        
        speaker_id = segment.get("dominant_speaker", "unknown")
        confidence = segment.get("confidence", 0)
        text = segment.get("text", "").lower()
        
        if self.ai_mode == "speaker-focus":
            if speaker_id != "unknown" and speaker_id != self.current_speaker:
                camera = self.whisper.diarizer.speaker_camera_map.get(speaker_id, 0)
                if self.obs.switch_to_camera(camera):
                    self.current_speaker = speaker_id
                    self.last_switch_time = now
                    await self.broadcast({
                        "type": "camera_switched",
                        "camera": camera,
                        "speaker": self.whisper.diarizer.speaker_names.get(speaker_id, "Unknown"),
                        "reason": "speaker_change"
                    })
        
        elif self.ai_mode == "voice-activity":
            if speaker_id != "unknown" and confidence > 0.6:
                camera = self.whisper.diarizer.speaker_camera_map.get(speaker_id, 0)
                if self.obs.switch_to_camera(camera):
                    self.last_switch_time = now
                    await self.broadcast({
                        "type": "camera_switched",
                        "camera": camera,
                        "reason": "voice_activity"
                    })
        
        elif self.ai_mode == "reaction-cam":
            # Detect excitement/reactions
            excitement = [("wow", "omg", "what", "no way", "amazing", "incredible", "!")]
            if any(ind in text for ind in excitement):
                reaction_cam = max(0, len(self.obs.camera_sources) - 1)
                if self.obs.switch_to_camera(reaction_cam):
                    self.last_switch_time = now
                    await self.broadcast({
                        "type": "camera_switched",
                        "camera": reaction_cam,
                        "reason": "reaction_detected"
                    })
                    
                    # Return after delay
                    await asyncio.sleep(3)
                    if self.current_speaker:
                        camera = self.whisper.diarizer.speaker_camera_map.get(self.current_speaker, 0)
                        self.obs.switch_to_camera(camera)
    
    async def calibrate_speaker(self, data: dict):
        """Calibrate speaker with audio sample"""
        speaker_id = data["speaker_id"]
        audio_bytes = bytes(data["audio_data"])
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        self.whisper.register_speaker_sample(speaker_id, audio_np)
    
    async def send_to_client(self, websocket: WebSocketServerProtocol, message: dict):
        """Send message to specific client"""
        try:
            await websocket.send(json.dumps(message))
        except:
            pass
    
    async def broadcast(self, message: dict):
        """Broadcast to all clients"""
        if self.clients:
            await asyncio.gather(*[
                client.send(json.dumps(message))
                for client in self.clients
            ], return_exceptions=True)
    
    async def start(self, host: str = "0.0.0.0", port: int = 8765):
        """Start server"""
        print(f"Starting PodCasteer Whisper Server on {host}:{port}")
        await self.initialize_whisper()
        
        async with websockets.serve(self.handle_client, host, port):
            print(f"Server running at ws://{host}:{port}")
            await asyncio.Future()


if __name__ == "__main__":
    server = PodCasteerServer()
    asyncio.run(server.start())
