# ROCKY AI 🤖

A Jarvis-style voice assistant for Windows, powered by Claude.

Talk to your computer — Rocky listens through your microphone, thinks with
Claude (via the Claude Agent SDK), speaks back with a natural neural voice,
and can **actually control your PC**: open apps, run commands, manage files,
and search the web.

## How it works

| Part  | Tech |
|-------|------|
| Ears  | `sounddevice` mic capture + Google speech recognition |
| Brain | Claude Agent SDK — rides your existing Claude Code login, no API key needed |
| Voice | Microsoft Edge neural TTS (`edge-tts`), offline SAPI fallback |
| Hands | Claude Code tools: Bash/PowerShell, file read/write/edit, web search |

## Setup

1. Install [Claude Code](https://claude.com/claude-code) and log in (`claude` in a terminal).
2. Install Python 3.10+ and the dependencies:

   ```
   pip install -r requirements.txt
   ```

## Run

```
python rocky.py           # voice mode: press Enter, then speak
python rocky.py --text    # type instead of talking
python rocky.py --safe    # read-only mode (Rocky can't change anything)
python rocky.py --mute    # no spoken replies
```

Say **"goodbye"** to shut Rocky down.

## Example commands

- "Open Chrome and search for the weather in Melbourne"
- "Create a folder called invoices on my desktop"
- "What's using the most memory on my PC right now?"
- "Summarise the readme file in my projects folder"

## Safety note

By default Rocky can run commands and edit files (that's the point of a
Jarvis). If you want a look-but-don't-touch assistant, run with `--safe`.
