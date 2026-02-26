#!/usr/bin/env python3
"""PodCasteer OBS Agent

Run this on the machine that has OBS.
It connects OUT to the PodCasteer server and performs OBS WebSocket actions locally.

Env vars:
- PODCASTEER_SERVER_URL (default: ws://localhost:8765)
- PODCASTEER_AGENT_NAME (default: obs-agent)

Example:
  PODCASTEER_SERVER_URL=ws://<server-ip>:8765 python3 obs_agent.py

Then in the PodCasteer UI, use obs_connect as usual. The server will relay.
"""

import asyncio
import json
import os
import time
from typing import List

import websockets

from obswebsocket import obsws, requests as obs_requests


class OBSSwitcher:
    def __init__(self):
        self.ws = None
        self.scene_name = "Scene"
        self.camera_sources: List[str] = []
        self.connected = False

    def connect(self, host: str = "localhost", port: int = 4455, password: str = "") -> None:
        self.ws = obsws(host, int(port), password)
        self.ws.connect()
        self.connected = True

    def disconnect(self) -> None:
        if self.ws:
            try:
                self.ws.disconnect()
            finally:
                self.ws = None
                self.connected = False

    def set_scene_name(self, name: str) -> None:
        self.scene_name = name

    def set_camera_sources(self, sources: List[str]) -> None:
        self.camera_sources = sources

    def switch_to_camera(self, index: int) -> bool:
        if not self.connected or index < 0 or index >= len(self.camera_sources):
            return False

        target_source = self.camera_sources[index]
        scene_items = self.ws.call(obs_requests.GetSceneItemList(sceneName=self.scene_name))

        for item in scene_items.getSceneItems():
            source_name = item["sourceName"]
            should_be_visible = source_name == target_source
            self.ws.call(
                obs_requests.SetSceneItemEnabled(
                    sceneName=self.scene_name,
                    sceneItemId=item["sceneItemId"],
                    sceneItemEnabled=should_be_visible,
                )
            )

        return True


async def run_agent() -> None:
    server_url = os.getenv("PODCASTEER_SERVER_URL", "ws://localhost:8765")
    agent_name = os.getenv("PODCASTEER_AGENT_NAME", "obs-agent")

    obs = OBSSwitcher()
    backoff = 1

    while True:
        try:
            async with websockets.connect(server_url) as ws:
                backoff = 1
                await ws.send(json.dumps({"type": "register_obs_agent", "name": agent_name}))

                async for raw in ws:
                    msg = json.loads(raw)
                    msg_type = msg.get("type")

                    if msg_type == "obs_connect":
                        request_id = msg.get("request_id")
                        try:
                            obs.connect(
                                host=msg.get("host", "localhost"),
                                port=msg.get("port", 4455),
                                password=msg.get("password", ""),
                            )
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "obs_status",
                                        "request_id": request_id,
                                        "connected": True,
                                    }
                                )
                            )
                        except Exception as e:
                            obs.disconnect()
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "obs_status",
                                        "request_id": request_id,
                                        "connected": False,
                                        "error": str(e),
                                    }
                                )
                            )

                    elif msg_type == "config":
                        if "scene_name" in msg:
                            obs.set_scene_name(msg["scene_name"])
                        if "camera_sources" in msg:
                            obs.set_camera_sources(msg["camera_sources"])

                    elif msg_type == "switch_camera":
                        try:
                            obs.switch_to_camera(int(msg.get("camera_index", 0)))
                        except Exception:
                            # keep agent alive even if switch fails
                            pass

        except asyncio.CancelledError:
            raise
        except Exception:
            # reconnect with backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == "__main__":
    asyncio.run(run_agent())
