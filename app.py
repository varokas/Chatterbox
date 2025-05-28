import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


model = ChatterboxTTS.from_pretrained(DEVICE)

def generate(text, audio_prompt_path, exaggeration, pace, temperature, seed_num, cfg_weight):
    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        pace=pace,
        temperature=temperature,
        cfg_weight=cfg_weight,
    )
    return model.sr, wav.squeeze(0).numpy()


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(value="What does the fox say?", label="Text to synthesize")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)


            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)
                pace = gr.Slider(0.8, 1.2, step=.01, label="pace", value=1)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate,
        inputs=[
            text,
            ref_wav,
            exaggeration,
            pace,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.launch()
