"""
main.py — LiveKit Voice AI Agent entry point (LiveKit Agents 1.5.0).
Armenian bank customer support agent for deposits, credits, and branches.
"""

import asyncio
import logging
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero
from src.config import SYSTEM_PROMPT, TTS_VOICE
from src.llm import run_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArmenianBankAgent(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)
        self._history = []

    async def on_enter(self):
        await self.session.say(
            "Բարև ձեզ։ Ես հայկական բանկերի AI օգնականն եմ։ "
            "Կարող եմ պատասխանել հարցերին ավանդների, վարկերի և մասնաճյուղերի վերաբերյալ։ "
            "Ինչո՞վ կարող եմ օգնել։",
            allow_interruptions=True,
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        user_text = new_message.text_content

        print(f"[DEBUG] Received: '{user_text}'")

        if not user_text or not user_text.strip():
            return

        if self._history and self._history[-2].get("content") == user_text:
            return

        logger.info(f"[agent] User: {user_text}")

        loop = asyncio.get_event_loop()
        answer, is_on_topic = await loop.run_in_executor(
            None,
            lambda: run_rag_pipeline(user_text, self._history),
        )

        logger.info(f"[agent] Answer: {answer}")

        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > 10:
            self._history = self._history[-10:]

        # turn_ctx.truncate(0)
        await self.session.say(answer, allow_interruptions=True)


def prewarm(proc: JobProcess):
    from src.rag.store import get_model
    get_model()
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    await ctx.wait_for_participant()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],  # prewarm-ից
        stt=openai.STT(
            model="gpt-4o-transcribe",
            language="hy",
            prompt="Արارատ բанк, Ամериа банк, ԱԿБА банк, авандный, варк, токос, масначяух, Արարատ բանկ, Ամերիա բանկ, ԱԿԲԱ բանկ, ավանդ, վարկ, տոկոս, մասնաճյուղ",
        ),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="tts-1", voice=TTS_VOICE, speed=0.85),
    )

    await session.start(
        room=ctx.room,
        agent=ArmenianBankAgent(),
    )

    await asyncio.sleep(3600)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=120.0,
        )
    )