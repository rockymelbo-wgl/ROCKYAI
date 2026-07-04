"""
ROCKY AI — a Jarvis-style voice assistant for your PC.

Ears  : your microphone (sounddevice) + Google speech recognition
Brain : Claude, via the Claude Agent SDK (uses your Claude Code login —
        no API key needed). Rocky can run commands, open apps, manage
        files, and search the web.
Voice : Microsoft Edge neural text-to-speech (falls back to offline
        Windows SAPI voice if there's no internet).

Usage:
    python rocky.py           voice mode (press Enter, then speak)
    python rocky.py --text    type instead of talking
    python rocky.py --safe    read-only mode (Rocky can't change anything)
"""

import argparse
import asyncio
import io
import re
import sys

import numpy as np
import sounddevice as sd

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

# ---------------------------------------------------------------- config

SAMPLE_RATE = 16_000
SILENCE_SECONDS = 1.3      # stop recording after this much quiet
MAX_RECORD_SECONDS = 20
ENERGY_THRESHOLD = 0.012   # RMS level that counts as speech

TTS_VOICE = "en-GB-RyanNeural"   # change to en-US-GuyNeural, en-AU-WilliamNeural, ...

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

SAFE_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]
FULL_TOOLS = SAFE_TOOLS + ["Bash", "Write", "Edit"]

EXIT_WORDS = {"exit", "quit", "goodbye", "good bye", "shut down", "shutdown rocky"}

# ---------------------------------------------------------------- voice out


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " code omitted ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_#>]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


async def speak(text: str) -> None:
    text = _strip_markdown(text)
    if not text:
        return
    try:
        import edge_tts
        import soundfile as sf

        communicate = edge_tts.Communicate(text, TTS_VOICE)
        mp3 = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3 += chunk["data"]
        data, rate = sf.read(io.BytesIO(mp3), dtype="float32")
        sd.play(data, rate)
        sd.wait()
    except Exception:
        # offline fallback: Windows built-in SAPI voice
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"  (TTS unavailable: {e})")


# ---------------------------------------------------------------- voice in


def record_until_silence() -> bytes | None:
    """Record from the default mic until the speaker goes quiet."""
    frames: list[np.ndarray] = []
    speech_started = False
    silent_chunks = 0
    chunk = int(SAMPLE_RATE * 0.1)  # 100 ms blocks
    silence_limit = int(SILENCE_SECONDS / 0.1)
    max_chunks = int(MAX_RECORD_SECONDS / 0.1)

    print("  listening... (speak now)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=chunk) as stream:
        for _ in range(max_chunks):
            block, _overflow = stream.read(chunk)
            frames.append(block.copy())
            rms = float(np.sqrt(np.mean((block.astype(np.float32) / 32768.0) ** 2)))
            if rms > ENERGY_THRESHOLD:
                speech_started = True
                silent_chunks = 0
            elif speech_started:
                silent_chunks += 1
                if silent_chunks >= silence_limit:
                    break
    if not speech_started:
        return None
    return np.concatenate(frames).tobytes()


def transcribe(raw_audio: bytes) -> str | None:
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    audio = sr.AudioData(raw_audio, SAMPLE_RATE, 2)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"  (speech service error: {e})")
        return None


# ---------------------------------------------------------------- brain


async def ask_rocky(client: ClaudeSDKClient, prompt: str) -> str:
    await client.query(prompt)
    reply_parts: list[str] = []
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"\nROCKY: {block.text}")
                    reply_parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    print(f"  [{block.name}...]")
        elif isinstance(message, ResultMessage) and message.is_error:
            reply_parts.append("Something went wrong with that request, sir.")
    return " ".join(reply_parts[-2:]) if reply_parts else "Done."


# ---------------------------------------------------------------- main loop


async def main() -> None:
    parser = argparse.ArgumentParser(description="ROCKY AI voice assistant")
    parser.add_argument("--text", action="store_true", help="type instead of speak")
    parser.add_argument("--safe", action="store_true", help="read-only tools")
    parser.add_argument("--mute", action="store_true", help="no spoken replies")
    args = parser.parse_args()

    tools = SAFE_TOOLS if args.safe else FULL_TOOLS
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=tools,
        permission_mode="acceptEdits",
    )

    print("=" * 52)
    print("  ROCKY AI online.")
    print(f"  mode: {'text' if args.text else 'voice'}"
          f"{' | SAFE (read-only)' if args.safe else ''}")
    print("  say or type 'goodbye' to shut down.")
    print("=" * 52)

    async with ClaudeSDKClient(options=options) as client:
        greeting = "Rocky A.I. online. How can I help you, sir?"
        print(f"\nROCKY: {greeting}")
        if not args.mute:
            await speak(greeting)

        while True:
            if args.text:
                try:
                    user_input = input("\nYOU: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
            else:
                try:
                    input("\n[press Enter to talk] ")
                except (EOFError, KeyboardInterrupt):
                    break
                raw = record_until_silence()
                if raw is None:
                    print("  (didn't catch anything)")
                    continue
                user_input = transcribe(raw) or ""
                if not user_input:
                    print("  (couldn't understand that — try again)")
                    continue
                print(f"YOU: {user_input}")

            if not user_input:
                continue
            if user_input.lower().strip(" .!") in EXIT_WORDS:
                farewell = "Powering down. Goodbye, sir."
                print(f"\nROCKY: {farewell}")
                if not args.mute:
                    await speak(farewell)
                break

            reply = await ask_rocky(client, user_input)
            if not args.mute:
                await speak(reply)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nROCKY: Offline.")
