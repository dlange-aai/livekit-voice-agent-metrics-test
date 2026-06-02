"""Real-room LiveKit agent that captures EOU metrics in the LiveKit Cloud
Agent insights dashboard.

Runs a full STT->LLM->TTS pipeline so both the user-turn EOU metrics
(transcription_delay, end_of_turn_delay) and the assistant-turn metrics
(llm_node_ttft, tts_node_ttfb, e2e_latency) land on the trace and show up under
Sessions -> Agent insights. Direct provider plugins throughout: AssemblyAI STT
(u3-rt-pro), Anthropic LLM, Cartesia TTS.

Run it:

  uv run agent.py dev
  # then open cloud.livekit.io, Launch Console, and talk (audio = your mic).

`console` mode does NOT connect to Cloud (mocked room) and won't appear in the
dashboard -- use `dev`.

Requires in .env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
ASSEMBLYAI_API_KEY, ANTHROPIC_API_KEY, CARTESIA_API_KEY -- and "Agent
observability" enabled (Settings -> Data and privacy).
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, TurnHandlingOptions, WorkerOptions, cli
from livekit.agents.metrics import LLMModelUsage, STTModelUsage, TTSModelUsage
from livekit.plugins import anthropic, assemblyai, cartesia, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logger = logging.getLogger("metrics-agent")


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 1000:.0f} ms"


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _p50(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[len(s) // 2]


def _print_metrics_summary(session: AgentSession) -> None:
    # Per-turn latency lives on each ChatMessage's `.metrics` (a MetricsReport
    # dict) -- the non-deprecated replacement for the `metrics_collected` event.
    # User messages carry transcription_delay / end_of_turn_delay /
    # on_user_turn_completed_delay; assistant messages carry llm_node_ttft /
    # tts_node_ttfb / e2e_latency. Walk the history pairing each user turn with
    # the assistant reply that follows it (the opening greeting is assistant-only).
    turns: list[dict[str, Any]] = []
    for msg in session.history.items:
        role = getattr(msg, "role", None)
        if role == "user":
            turns.append({"user": msg, "assistant": None})
        elif role == "assistant":
            if turns and turns[-1]["assistant"] is None and turns[-1]["user"] is not None:
                turns[-1]["assistant"] = msg
            else:
                turns.append({"user": None, "assistant": msg})

    if not turns:
        logger.info("no metrics collected")
        return

    cols = (
        "turn",
        "td (transcription_delay)",
        "eou (end_of_turn_delay)",
        "on_user_turn_cb",
        "llm_ttft",
        "tts_ttfb",
        "e2e_latency",
    )
    numeric_cols = ("td", "eou", "on_user_turn_cb", "llm_ttft", "tts_ttfb", "e2e")
    numeric: dict[str, list[float]] = {k: [] for k in numeric_cols}

    def _take(key: str, v: float | None) -> str:
        if v is not None:
            numeric[key].append(v)
        return _fmt_ms(v)

    rows: list[tuple[str, ...]] = []
    user_turn = 0
    for t in turns:
        um = t["user"].metrics if t["user"] is not None else {}
        am = t["assistant"].metrics if t["assistant"] is not None else {}
        if t["user"] is not None:
            user_turn += 1
            label = str(user_turn)
        else:
            label = "greet"
        rows.append(
            (
                label,
                _take("td", um.get("transcription_delay")),
                _take("eou", um.get("end_of_turn_delay")),
                _take("on_user_turn_cb", um.get("on_user_turn_completed_delay")),
                _take("llm_ttft", am.get("llm_node_ttft")),
                _take("tts_ttfb", am.get("tts_node_ttfb")),
                _take("e2e", am.get("e2e_latency")),
            )
        )

    def _stat_row(label: str, fn: Callable[[list[float]], float | None]) -> tuple[str, ...]:
        return (label, *(_fmt_ms(fn(numeric[k])) for k in numeric_cols))

    stat_rows = [_stat_row("mean", _mean), _stat_row("p50", _p50)]

    widths = [max(len(r[c]) for r in [cols] + rows + stat_rows) for c in range(len(cols))]
    sep = "  "
    total_w = sum(widths) + len(sep) * (len(widths) - 1)
    print()
    print("=" * total_w)
    print("LiveKit session metrics (cross-reference vs Agent insights dashboard)")
    print("=" * total_w)
    print(sep.join(c.ljust(w) for c, w in zip(cols, widths)))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(sep.join(c.ljust(w) for c, w in zip(r, widths)))
    print(sep.join("=" * w for w in widths))
    for r in stat_rows:
        print(sep.join(c.ljust(w) for c, w in zip(r, widths)))
    print()
    print(f"(stats computed over {len(numeric['td'])} user-turns with EOU metrics)")

    # Aggregate token/audio usage via the non-deprecated `session.usage`
    # (the `session_usage_updated` surface), one entry per model/provider.
    usage = session.usage
    if usage.model_usage:
        print("\nsession usage:")
        for mu in usage.model_usage:
            tag = f"{mu.provider}/{mu.model}"
            if isinstance(mu, LLMModelUsage):
                print(f"  llm  {tag}: {mu.input_tokens} in / {mu.output_tokens} out tokens")
            elif isinstance(mu, STTModelUsage):
                print(f"  stt  {tag}: {mu.audio_duration:.1f}s audio")
            elif isinstance(mu, TTSModelUsage):
                print(f"  tts  {tag}: {mu.characters_count} chars / {mu.audio_duration:.1f}s audio")
            else:
                print(f"  {getattr(mu, 'type', '?')}  {tag}")
    print()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    # STT: AssemblyAI u3-rt-pro, constructor args only. In a Worker the turn
    # detector pulls its inference executor from the job context automatically.
    #
    # LLM + TTS don't affect transcription_delay (measured at user-turn
    # completion, upstream of the LLM) -- they just make the assistant-side
    # metrics appear in the timeline. Plugin defaults are fine here.
    session = AgentSession(
        stt=assemblyai.STT(
            model="u3-rt-pro",
            min_turn_silence=100,
            max_turn_silence=100,
            vad_threshold=0.3
        ),
        llm=anthropic.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(activation_threshold=0.3),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            endpointing={"min_delay": 0, "max_delay": 0},
        ),
    )

    # Read per-turn latency and usage off the session at shutdown instead of the
    # deprecated `metrics_collected` event: each ChatMessage in `session.history`
    # carries a `.metrics` report and `session.usage` aggregates token/audio
    # usage. This is the same data the LiveKit Cloud "Agent insights" dashboard
    # shows, so the printed table cross-references against it.
    async def _print_summary() -> None:
        _print_metrics_summary(session)

    ctx.add_shutdown_callback(_print_summary)

    # record=True uploads audio, transcript, traces (including the EOU metrics
    # enriched onto the user-turn span and the assistant-turn LLM/TTS metrics),
    # and logs to LiveKit Cloud. (Omitting it defers to the server-side setting.)
    await session.start(
        agent=Agent(
            instructions=(
                "You are a helpful voice assistant. Answer the user's question "
                "in one short sentence."
            )
        ),
        room=ctx.room,
        record=True,
    )

    # Greeting via `session.say()` (TTS only, no LLM): reads out the fixed
    # question script so the user just echoes each line back, one turn at a
    # time, giving the same set of turns every run. It produces no user-turn EOU
    # metric (those are tied to USER speech), so it doesn't pollute the numbers.
    await session.say(
        "Hi. When you're ready, read me these questions one at a time. "
        "One: What is the capital of France? "
        "Two: Tell me a fun fact about octopuses. "
        "Three: What does HTTP stand for? "
        "Four: Name a country that borders Brazil. "
        "Five: How many planets are in our solar system? "
        "Then say: Thanks, goodbye.",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
