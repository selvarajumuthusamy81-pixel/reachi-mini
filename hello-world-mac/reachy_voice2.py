import io
import time
import wave

import numpy as np
from scipy.signal import resample

from openai import OpenAI
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


client = OpenAI()

MODEL = "gpt-5.6-luna"


# ============================================================
# AUDIO
# ============================================================

def samples_to_wav(samples, sample_rate):
    samples = np.asarray(samples)

    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    buffer.seek(0)
    return buffer


def record_from_reachy(mini, seconds=5):
    print("\n🎤 Listening...")

    samples = []

    mini.media.start_recording()

    start = time.time()

    try:
        while time.time() - start < seconds:

            sample = mini.media.get_audio_sample()

            if sample is not None:
                samples.append(sample)

            time.sleep(0.005)

    finally:
        mini.media.stop_recording()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=0)

    sample_rate = mini.media.get_input_audio_samplerate()

    return samples_to_wav(audio, sample_rate)


def transcribe(audio_file):

    print("📝 Transcribing...")

    result = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=("reachy.wav", audio_file, "audio/wav"),
    )

    return result.text.strip()


# ============================================================
# EMOTIONS
# ============================================================

def greeting(mini):

    mini.goto_target(
        head=create_head_pose(pitch=-5, degrees=True),
        antennas=np.deg2rad([20, 20]),
        duration=0.5,
        method="cartoon",
    )

    mini.goto_target(
        antennas=np.deg2rad([-25, 25]),
        duration=0.3,
        method="cartoon",
    )

    mini.goto_target(
        antennas=np.deg2rad([25, -25]),
        duration=0.3,
        method="cartoon",
    )

    mini.goto_target(
        antennas=np.deg2rad([0, 0]),
        duration=0.4,
        method="cartoon",
    )


def happy(mini):

    print("😊 HAPPY")

    mini.goto_target(
        head=create_head_pose(pitch=-8, degrees=True),
        antennas=np.deg2rad([25, 25]),
        duration=0.5,
        method="cartoon",
    )

    mini.goto_target(
        head=create_head_pose(pitch=5, degrees=True),
        antennas=np.deg2rad([-10, -10]),
        duration=0.35,
        method="cartoon",
    )

    mini.goto_target(
        head=create_head_pose(pitch=-3, degrees=True),
        antennas=np.deg2rad([15, 15]),
        duration=0.35,
        method="cartoon",
    )


def surprised(mini):

    print("😮 SURPRISED")

    mini.goto_target(
        head=create_head_pose(pitch=-8, degrees=True),
        antennas=np.deg2rad([45, 45]),
        duration=0.4,
        method="cartoon",
    )

    time.sleep(0.5)

    mini.goto_target(
        head=create_head_pose(),
        antennas=np.deg2rad([10, 10]),
        duration=0.5,
        method="cartoon",
    )


def thinking(mini):

    print("🤔 THINKING")

    mini.goto_target(
        head=create_head_pose(
            yaw=-18,
            roll=8,
            degrees=True,
        ),
        antennas=np.deg2rad([10, -10]),
        duration=0.7,
        method="cartoon",
    )

    time.sleep(0.6)

    mini.goto_target(
        head=create_head_pose(
            yaw=18,
            roll=-8,
            degrees=True,
        ),
        antennas=np.deg2rad([-10, 10]),
        duration=0.7,
        method="cartoon",
    )

    time.sleep(0.6)

    mini.goto_target(
        head=create_head_pose(),
        antennas=np.deg2rad([0, 0]),
        duration=0.6,
        method="cartoon",
    )


def sad(mini):

    print("😢 SAD")

    mini.goto_target(
        head=create_head_pose(
            pitch=15,
            roll=-5,
            degrees=True,
        ),
        antennas=np.deg2rad([-20, -20]),
        duration=0.8,
        method="cartoon",
    )


def neutral(mini):

    mini.goto_target(
        head=create_head_pose(),
        antennas=np.deg2rad([0, 0]),
        duration=0.6,
        method="cartoon",
    )


def perform_emotion(mini, emotion):

    emotion = emotion.lower().strip()

    if emotion == "happy":
        happy(mini)

    elif emotion == "surprised":
        surprised(mini)

    elif emotion == "thinking":
        thinking(mini)

    elif emotion == "sad":
        sad(mini)

    elif emotion == "greeting":
        greeting(mini)

    else:
        neutral(mini)


# ============================================================
# OPENAI
# ============================================================

def ask_openai(text):

    print("🤖 Thinking...")

    prompt = f"""
You are Reachy, a friendly expressive robot.

The user said:

"{text}"

Respond with EXACTLY this format:

EMOTION: <one emotion>
RESPONSE: <what Reachy should say>

Allowed emotions:

happy
sad
surprised
thinking
greeting
neutral

Choose the emotion that naturally fits the conversation.

Keep the response conversational and fairly short.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    output = response.output_text.strip()

    emotion = "neutral"
    answer = output

    for line in output.splitlines():

        if line.upper().startswith("EMOTION:"):
            emotion = line.split(":", 1)[1].strip()

        elif line.upper().startswith("RESPONSE:"):
            answer = line.split(":", 1)[1].strip()

    return emotion, answer


# ============================================================
# SPEECH
# ============================================================

def speak_with_reachy(mini, text):

    print(f"🔊 Reachy: {text}")

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav",
    )

    wav_bytes = speech.read()

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:

        sample_rate = wav.getframerate()
        channels = wav.getnchannels()

        raw_audio = wav.readframes(wav.getnframes())

    audio = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    )

    audio = audio.reshape(-1, channels)

    audio = audio.astype(np.float32) / 32768.0

    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    output_rate = mini.media.get_output_audio_samplerate()

    if sample_rate != output_rate:

        new_length = int(
            len(audio) * output_rate / sample_rate
        )

        audio = resample(
            audio,
            new_length,
        )

    audio = audio.astype(np.float32).reshape(-1, 1)

    mini.media.start_playing()

    try:

        chunk_size = 1024

        for i in range(
            0,
            len(audio),
            chunk_size,
        ):

            chunk = audio[i:i + chunk_size]

            mini.media.push_audio_sample(chunk)

            time.sleep(
                len(chunk) / output_rate
            )

    finally:

        mini.media.stop_playing()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("======================================")
    print("   🤖 Reachy Expressive AI")
    print("======================================")
    print()

    with ReachyMini(
        media_backend="default"
    ) as mini:

        print("✅ Connected to Reachy Mini")
        print("💬 Expressive conversation started")
        print("Press Ctrl+C to exit.")
        print()

        while True:

            # 1. Listen
            audio_file = record_from_reachy(
                mini,
                seconds=5,
            )

            if audio_file is None:
                continue

            # 2. Speech → text
            text = transcribe(audio_file)

            if not text:
                continue

            print()
            print(f"You: {text}")

            # 3. AI decides emotion + response
            emotion, answer = ask_openai(text)

            print(f"🎭 Emotion: {emotion}")

            # 4. Robot expresses emotion
            perform_emotion(
                mini,
                emotion,
            )

            # 5. Robot speaks
            speak_with_reachy(
                mini,
                answer,
            )

            print()


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print("\n👋 Goodbye!")
