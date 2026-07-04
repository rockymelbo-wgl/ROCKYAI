"""
ROCKY AI — Web UI server.

Serves an animated robot interface in the browser and bridges it to the
Claude Agent SDK brain over a WebSocket. Speech-to-text happens in the
browser (Web Speech API); the spoken reply is synthesized server-side
with Edge neural TTS and streamed back as audio.

Run:
    python webui.py
then open http://localhost:8765
"""

import asyncio
import base64
import io
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

ROOT = Path(__file__).parent
PORT = 8765
TTS_VOICE = "en-GB-RyanNeural"

SYSTEM_PROMPT = """You are ROCKY, a Jarvis-style AI assistant running on the \
user's Windows PC. You are capable, loyal, and slightly witty — think Tony \
Stark's Jarvis. Address the user as "sir" occasionally, but don't overdo it.

You have real tools: you can run PowerShell/commands, read and write files, \
search the web, and open applications (use `start <app>` or `start <url>` \
via Bash for that). When the user asks you to do something on the computer, \
actually do it with your tools — don't just describe how.

Your replies are spoken aloud through text-to-speech, so:
- Keep responses SHORT and conversational — one to three sentences when possible.
- No markdown, no bullet lists, no code blocks in your final reply.
- After completing an action, confirm briefly, e.g. "Done, sir. Chrome is open."
- If a task produces long output, summarise it in a sentence or two."""

TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch", "Bash", "Write", "Edit"]

app = FastAPI(title="Rocky AI")


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " code omitted ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_#>]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


async def synthesize(text: str) -> str | None:
    """Return base64 mp3 of the spoken text, or None if TTS fails."""
    text = strip_markdown(text)
    if not text:
        return None
    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, TTS_VOICE)
        mp3 = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        return base64.b64encode(mp3).decode()
    except Exception:
        return None  # browser falls back to speechSynthesis


def transcribe_pcm(pcm16: bytes) -> str | None:
    """Speech-to-text on 16 kHz mono 16-bit PCM from the browser mic."""
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    audio = sr.AudioData(pcm16, 16_000, 2)
    try:
        return recognizer.recognize_google(audio)
    except Exception:
        return None


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=TOOLS,
        permission_mode="acceptEdits",
    )
    try:
        async with ClaudeSDKClient(options=options) as client:
            greeting = "Rocky A.I. online. How can I help you, sir?"
            await ws.send_json({
                "type": "reply",
                "text": greeting,
                "audio": await synthesize(greeting),
            })

            async def run_query(user_text: str) -> None:
                await ws.send_json({"type": "state", "value": "thinking"})
                reply_parts: list[str] = []
                try:
                    await client.query(user_text)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            for block in message.content:
                                if isinstance(block, TextBlock):
                                    reply_parts.append(block.text)
                                    await ws.send_json(
                                        {"type": "partial", "text": block.text})
                                elif isinstance(block, ToolUseBlock):
                                    await ws.send_json({
                                        "type": "state",
                                        "value": "working",
                                        "label": block.name,
                                    })
                        elif isinstance(message, ResultMessage) and message.is_error:
                            reply_parts.append(
                                "Something went wrong with that request, sir.")
                except Exception as e:
                    reply_parts = [f"I hit a problem, sir: {e}"]

                reply = " ".join(reply_parts[-2:]) if reply_parts else "Done, sir."
                await ws.send_json({
                    "type": "reply",
                    "text": reply,
                    "audio": await synthesize(reply),
                })

            while True:
                msg = await ws.receive_json()

                if msg.get("type") == "user_text":
                    user_text = str(msg.get("text", "")).strip()
                    if user_text:
                        await run_query(user_text)

                elif msg.get("type") == "user_audio":
                    # raw 16 kHz mono pcm16 from the browser mic, base64
                    await ws.send_json({"type": "state", "value": "thinking"})
                    try:
                        pcm = base64.b64decode(msg.get("pcm16", ""))
                    except Exception:
                        pcm = b""
                    text = None
                    if len(pcm) > 8000:  # ignore sub-quarter-second blips
                        text = await asyncio.to_thread(transcribe_pcm, pcm)
                    await ws.send_json({"type": "transcript", "text": text or ""})
                    if text:
                        await run_query(text)
                    else:
                        await ws.send_json({"type": "state", "value": "idle"})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    print(f"\n  ROCKY AI web interface -> http://localhost:{PORT}\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
