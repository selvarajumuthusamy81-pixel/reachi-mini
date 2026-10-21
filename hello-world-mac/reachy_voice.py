import io
import time
import wave

import numpy as np
from scipy.signal import resample

from openai import OpenAI
from reachy_mini import ReachyMini


client = OpenAI()


def samples_to_wav(samples, sample_rate):
    """Convert Reachy float32 audio samples to a WAV file in memory."""

    # Reachy gives us float32 samples in the range -1.0 .. 1.0
    samples = np.asarray(samples)

    # Convert stereo to mono
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    # Convert float32 -> 16-bit PCM
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
    """Record from Reachy's microphone."""

    print(f"\n🎤 Listening for {seconds} seconds...")

    samples = []

    mini.media.start_recording()

    # Give the microphone a moment to initialize.
    time.sleep(0.2)

    start = time.time()

    while time.time() - start < seconds:
        sample = mini.media.get_audio_sample()

        if sample is not None:
            samples.append(sample)

        time.sleep(0.005)

    mini.media.stop_recording()

    if not samples:
        raise RuntimeError("No microphone audio received.")

    audio = np.concatenate(samples, axis=0)

    sample_rate = mini.media.get_input_audio_samplerate()

    print(f"Recorded {len(audio) / sample_rate:.1f} seconds")

    return samples_to_wav(audio, sample_rate)


def transcribe(audio_file):
    """Send microphone audio to OpenAI speech-to-text."""

    print("📝 Transcribing...")

    result = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=("reachy.wav", audio_file, "audio/wav"),
    )

    return result.text


def ask_openai(text):
    """Send the transcription to the conversation model."""

    print("🤖 Thinking...")

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=text,
    )

    return response.output_text


def speak_with_reachy(mini, text):
    """Generate speech with OpenAI and play it through Reachy's speaker."""

    print("🔊 Speaking:", text)

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="wav",
    )

    wav_bytes = speech.read()

    # Read the WAV returned by OpenAI.
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        raw_audio = wav.readframes(wav.getnframes())

    audio = np.frombuffer(raw_audio, dtype=np.int16)
    audio = audio.reshape(-1, channels)

    # Convert to float32 for Reachy.
    audio = audio.astype(np.float32) / 32768.0

    # Convert to mono if necessary.
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    # Reachy expects 16 kHz.
    output_rate = mini.media.get_output_audio_samplerate()

    if sample_rate != output_rate:
        new_length = int(len(audio) * output_rate / sample_rate)
        audio = resample(audio, new_length)

    audio = audio.astype(np.float32).reshape(-1, 1)

    mini.media.start_playing()

    mini.media.push_audio_sample(audio)

    # Wait for playback to finish.
    duration = len(audio) / output_rate
    time.sleep(duration + 0.1)

    mini.media.stop_playing()


def main():
    print("🤖 Reachy Mini + OpenAI Voice")

    with ReachyMini(media_backend="default") as mini:

        print("Connected to Reachy Mini!")
        print("Press Ctrl+C to stop.")

        while True:

            # 1. Reachy microphone
            audio_file = record_from_reachy(mini, seconds=5)

            # 2. Speech → text
            text = transcribe(audio_file)

            if not text.strip():
                print("I didn't hear anything.")
                continue

            print("You:", text)

            # 3. Text → GPT
            answer = ask_openai(text)

            print("AI:", answer)

            # 4. GPT → speech → Reachy speaker
            speak_with_reachy(mini, answer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye!")
