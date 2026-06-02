# LiveKit voice agent — metrics test

A real-room LiveKit voice agent (AssemblyAI STT → Anthropic LLM → Cartesia TTS)
that captures the per-turn latency metrics LiveKit emits and prints them at
session end, so you can cross-reference against the **Agent insights** dashboard
in LiveKit Cloud.

It runs a full STT→LLM→TTS pipeline so both the user-turn EOU metrics
(`transcription_delay`, `end_of_turn_delay`) and the assistant-turn metrics
(`llm_node_ttft`, `tts_node_ttfb`, `e2e_latency`) land on the trace.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python package manager). Install on macOS:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  `uv` provisions Python 3.10+ automatically — no separate Python install needed.
- A **LiveKit Cloud** project (free tier is fine): https://cloud.livekit.io
- API keys for **AssemblyAI**, **Anthropic**, and **Cartesia**.

## Setup

1. **Configure secrets.** Copy the example env file and fill in all six values:
   ```sh
   cp .env.example .env
   ```
   ```sh
   # .env
   LIVEKIT_URL=wss://<your-project>.livekit.cloud
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   ASSEMBLYAI_API_KEY=...
   ANTHROPIC_API_KEY=...
   CARTESIA_API_KEY=...
   ```
   The `LIVEKIT_*` values come from your LiveKit Cloud project settings (API keys).

2. **Install dependencies** into a project virtualenv:
   ```sh
   uv sync
   ```

3. **Pre-download the model files** (Silero VAD + the turn-detector model) so the
   first session doesn't stall:
   ```sh
   uv run agent.py download-files
   ```

4. **Enable observability** in your LiveKit Cloud project so the metrics reach the
   dashboard: **Settings → Data and privacy → Agent observability**.

## Run

```sh
uv run agent.py dev
```

Then open https://cloud.livekit.io → your project → **Launch Console**, allow the
mic, and talk. The agent greets with a fixed five-question script — read each
question back, one at a time, then say "Thanks, goodbye."

When the session ends, a metrics table and a usage summary print to the worker's
stdout, e.g.:

```
turn   td (transcription_delay)  eou (end_of_turn_delay)  on_user_turn_cb  llm_ttft  tts_ttfb  e2e_latency
1      295 ms                    647 ms                   0 ms             1021 ms   321 ms    1638 ms
...
mean   196 ms                    563 ms                   0 ms             1028 ms   409 ms    1552 ms
p50    270 ms                    541 ms                   0 ms             1021 ms   321 ms    1687 ms

session usage:
  llm  api.anthropic.com/claude-sonnet-4-6: 989 in / 95 out tokens
  tts  Cartesia/sonic-3: 544 chars / 34.9s audio
  stt  AssemblyAI/u3-rt-pro: 84.8s audio
```

The same data appears under **Sessions → Agent insights** in the dashboard.

> **Note:** `uv run agent.py console` runs a local, mocked room that does **not**
> connect to Cloud and won't appear in the dashboard — use `dev`.

## Metrics

| Column            | Source (`ChatMessage.metrics`)   | Meaning |
|-------------------|----------------------------------|---------|
| `td`              | `transcription_delay`            | STT delay finalizing the user transcript |
| `eou`             | `end_of_turn_delay`              | End-of-utterance / turn-detection delay |
| `on_user_turn_cb` | `on_user_turn_completed_delay`   | Time spent in the user-turn-completed callback |
| `llm_ttft`        | `llm_node_ttft`                  | LLM time-to-first-token |
| `tts_ttfb`        | `tts_node_ttfb`                  | TTS time-to-first-byte |
| `e2e_latency`     | `e2e_latency`                    | End-to-end user-stop → agent-audio latency |

Per-turn metrics are read from `session.history` and usage from `session.usage`
at shutdown (the non-deprecated APIs in livekit-agents 1.5+).

## Configuration

Everything lives in `agent.py`:
- **STT** — `assemblyai.STT(model="u3-rt-pro", …)`; turn-silence and VAD thresholds tuned low for latency measurement.
- **Turn detection** — LiveKit `MultilingualModel()` with `endpointing` delays set to `0` (strips added latency for the benchmark; raise these for a production agent).
- **LLM / TTS** — `anthropic.LLM()` and `cartesia.TTS()` plugin defaults.

## Notes

- `.env` holds real secrets — keep it out of version control (commit only `.env.example`).
- If `uv` resolves an older release than expected, your environment may be
  suppressing very recent versions for "minimum package age"; pass
  `--safe-chain-skip-minimum-package-age` to `uv sync` / `uv lock` to allow them.
