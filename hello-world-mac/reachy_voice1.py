import io
import time
import wave

import numpy as np
from scipy.signal import resample

from openai import OpenAI
from reachy_mini import ReachyMini


client = OpenAI()

MODEL = "gpt-5.6-luna"

SAMPLE_RATE = 16000

# Audio detection settings
SILENCE_THRESHOLD = 0.015
START_TIMEOUT = 10.0
SILENCE_DURATION = 1.2
MAX_RECORDING_SECONDS = 15.0


def samples_to_wav(samples, sample_rate):
    """Convert Reachy audio samples to an in-memory WAV file."""

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


def audio_level(sample):
    """Calculate the volume of an audio sample."""

    sample = np.asarray(sample)

    if sample.ndim > 1:
        sample = sample.mean(axis=1)

    return float(np.sqrt(np.mean(sample ** 2)))


def record_until_silence(mini):
    """
    Wait for the user to speak, then record until they stop talking.
    """

    print("\n🎤 Listening...")

    samples = []

    mini.media.start_recording()

    start_time = time.time()
    speech_started = False
    silence_start = None

    try:
        while True:

            sample = mini.media.get_audio_sample()

            if sample is None:
                time.sleep(0.005)
                continue

            samples.append(sample)

            level = audio_level(sample)

            now = time.time()

            # --------------------------------
            # Waiting for the user to speak
            # --------------------------------

            if not speech_started:

                if level > SILENCE_THRESHOLD:
                    speech_started = True
                    print("🗣️ I hear you...")

                elif now - start_time > START_TIMEOUT:
                    print("⌛ Nobody spoke.")
                    return None

            # --------------------------------
            # User is speaking
            # --------------------------------

            else:

                if level < SILENCE_THRESHOLD:

                    if silence_start is None:
                        silence_start = now

                    elif now - silence_start >= SILENCE_DURATION:
                        print("✅ Finished listening.")
                        break

                else:
                    silence_start = None

            # Safety limit
            if now - start_time > MAX_RECORDING_SECONDS:
                print("⏱️ Maximum recording time reached.")
                break

            time.sleep(0.005)

    finally:
        mini.media.stop_recording()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=0)

    sample_rate = mini.media.get_input_audio_samplerate()

    duration = len(audio) / sample_rate

    print(f"Recorded {duration:.1f} seconds")

    return samples_to_wav(audio, sample_rate)


def transcribe(audio_file):
    """Convert speech to text."""

    print("📝 Transcribing...")

    result = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=("reachy.wav", audio_file, "audio/wav"),
    )

    return result.text.strip()


def ask_openai(text, previous_response_id=None):
    """Send text to OpenAI while preserving conversation context."""

    print("🤖 Thinking...")

    kwargs = {
        "model": MODEL,
        "input": text,
    }

    if previous_response_id:
        kwargs["previous_response_id"] = previous_response_id

    response = client.responses.create(**kwargs)

    return response.output_text.strip(), response.id


def speak_with_reachy(mini, text):
    """Generate speech and play it through Reachy's speaker."""

    print("🔊 Reachy:", text)

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

    audio = np.frombuffer(raw_audio, dtype=np.int16)

    audio = audio.reshape(-1, channels)

    audio = audio.astype(np.float32) / 32768.0

    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    output_rate = mini.media.get_output_audio_samplerate()

    if sample_rate != output_rate:

        new_length = int(
            len(audio) * output_rate / sample_rate
        )

        audio = resample(audio, new_length)

    audio = audio.astype(np.float32).reshape(-1, 1)

    mini.media.start_playing()

    try:
        # Send audio in chunks rather than all at once.
        chunk_size = 1024

        for i in range(0, len(audio), chunk_size):

            chunk = audio[i:i + chunk_size]

            mini.media.push_audio_sample(chunk)

            time.sleep(len(chunk) / output_rate)

    finally:
        mini.media.stop_playing()


def main():

    print()
    print("===================================")
    print("   🤖 Reachy Mini AI Conversation")
    print("===================================")
    print()

    previous_response_id = None

    with ReachyMini(media_backend="default") as mini:

        print("✅ Connected to Reachy Mini")
        print("💬 Conversation started")
        print("Press Ctrl+C to exit.")
        print()

        while True:

            # -------------------------------
            # 1. Listen
            # -------------------------------

            audio_file = record_until_silence(mini)

            if audio_file is None:
                continue

            # -------------------------------
            # 2. Speech → text
            # -------------------------------

            text = transcribe(audio_file)

            if not text:
                continue

            print()
            print("You:", text)

            # -------------------------------
            # 3. Ask OpenAI
            # -------------------------------

            answer, previous_response_id = ask_openai(
                text,
                previous_response_id,
            )

            print()

            # -------------------------------
            # 4. Speak answer
            # -------------------------------

            speak_with_reachy(mini, answer)

            print()


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
