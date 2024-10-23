# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import re
import tempfile

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from pydub import AudioSegment

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from model import DiT, UNetT
from model.utils import save_spectrogram
from model.utils_infer import load_vocoder, load_model, preprocess_ref_audio_text, infer_process, remove_silence_for_generated_wav

vocos = load_vocoder()

# load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(DiT, F5TTS_model_cfg, str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")))

E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
E2TTS_ema_model = load_model(UNetT, E2TTS_model_cfg, str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors")))


@gpu_decorator
def infer(ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=gr.Info)

    if model == "F5-TTS":
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        ema_model = E2TTS_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=gr.Info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


@gpu_decorator
def generate_podcast(script, speaker1_name, ref_audio1, ref_text1, speaker2_name, ref_audio2, ref_text2, speaker3_name, ref_audio3, ref_text3, model, remove_silence):
    # Split the script into speaker blocks
    speaker_pattern = re.compile(f"^({re.escape(speaker1_name)}|{re.escape(speaker2_name)}|{re.escape(speaker3_name)}):", re.MULTILINE)
    speaker_blocks = speaker_pattern.split(script)[1:]  # Skip the first empty element

    generated_audio_segments = []

    for i in range(0, len(speaker_blocks), 2):
        speaker = speaker_blocks[i]
        text = speaker_blocks[i + 1].strip()

        # Determine which speaker is talking
        if speaker == speaker1_name:
            ref_audio = ref_audio1
            ref_text = ref_text1
        elif speaker == speaker2_name:
            ref_audio = ref_audio2
            ref_text = ref_text2
        elif speaker == speaker3_name:
            ref_audio = ref_audio3
            ref_text = ref_text3
        else:
            continue  # Skip if the speaker is none of the three

        # Generate audio for this block
        audio, _ = infer(ref_audio, ref_text, text, model, remove_silence)

        # Convert the generated audio to a numpy array
        sr, audio_data = audio

        # Save the audio data as a WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sr)
            audio_segment = AudioSegment.from_wav(temp_file.name)

        generated_audio_segments.append(audio_segment)

        # Add a short pause between speakers
        pause = AudioSegment.silent(duration=500)  # 500ms pause
        generated_audio_segments.append(pause)

    # Concatenate all audio segments
    final_podcast = sum(generated_audio_segments)

    # Export the final podcast
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        podcast_path = temp_file.name
        final_podcast.export(podcast_path, format="wav")

    return podcast_path


with gr.Blocks() as app_podcast:
    gr.Markdown("# Podcast Generation")

    # Speaker 1
    speaker1_name = gr.Textbox(label="Speaker 1 Name")
    ref_audio_input1 = gr.Audio(label="Reference Audio (Speaker 1)", type="filepath")
    ref_text_input1 = gr.Textbox(label="Reference Text (Speaker 1)", lines=2)

    # Speaker 2
    speaker2_name = gr.Textbox(label="Speaker 2 Name")
    ref_audio_input2 = gr.Audio(label="Reference Audio (Speaker 2)", type="filepath")
    ref_text_input2 = gr.Textbox(label="Reference Text (Speaker 2)", lines=2)

    # Speaker 3 (New)
    speaker3_name = gr.Textbox(label="Speaker 3 Name")
    ref_audio_input3 = gr.Audio(label="Reference Audio (Speaker 3)", type="filepath")
    ref_text_input3 = gr.Textbox(label="Reference Text (Speaker 3)", lines=2)

    script_input = gr.Textbox(
        label="Podcast Script",
        lines=10,
        placeholder="Enter the script with speaker names at the start of each block, e.g.:\nSean: How did you start studying...\n\nMeghan: I came to my interest in technology...\n\nAlex: That's fascinating. Can you elaborate...",
    )

    podcast_model_choice = gr.Radio(choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS")
    podcast_remove_silence = gr.Checkbox(label="Remove Silences", value=True)
    generate_podcast_btn = gr.Button("Generate Podcast", variant="primary")
    podcast_output = gr.Audio(label="Generated Podcast")

    def podcast_generation(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, speaker3, ref_audio3, ref_text3, model, remove_silence):
        return generate_podcast(script, speaker1, ref_audio1, ref_text1, speaker2, ref_audio2, ref_text2, speaker3, ref_audio3, ref_text3, model, remove_silence)

    generate_podcast_btn.click(
        podcast_generation,
        inputs=[
            script_input,
            speaker1_name,
            ref_audio_input1,
            ref_text_input1,
            speaker2_name,
            ref_audio_input2,
            ref_text_input2,
            speaker3_name,
            ref_audio_input3,
            ref_text_input3,
            podcast_model_choice,
            podcast_remove_silence,
        ],
        outputs=podcast_output,
    )

with gr.Blocks() as app:
    gr.Markdown("# E2/F5 TTS - Podcast with 3 Speakers")
    app_podcast.launch()
