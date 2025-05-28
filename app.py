import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr
import spaces # <<< IMPORT THIS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}") # Good to log this

# Global model variable to load only once if not using gr.State for model object
# global_model = None

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda": # Only seed cuda if available
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# Optional: Decorate model loading if it's done on first use within a GPU function
# However, it's often better to load the model once globally or manage with gr.State
# and ensure the function CALLING the model is decorated.

@spaces.GPU # <<< ADD THIS DECORATOR
def generate(model_obj, text, audio_prompt_path, exaggeration, pace, temperature, seed_num, cfgw):
    # It's better to load the model once, perhaps when the gr.State is initialized
    # or globally, rather than checking `model_obj is None` on every call.
    # For ZeroGPU, the decorated function handles the GPU context.
    # Let's assume model_obj is passed correctly and is already on DEVICE
    # or will be moved to DEVICE by ChatterboxTTS internally.

    if model_obj is None:
        print("Model is None, attempting to load...")
        # This load should ideally happen on DEVICE and be efficient.
        # If ChatterboxTTS.from_pretrained(DEVICE) is slow,
        # this will happen inside the GPU-allocated time.
        model_obj = ChatterboxTTS.from_pretrained(DEVICE)
        print(f"Model loaded on device: {model_obj.device if hasattr(model_obj, 'device') else 'unknown'}")


    if seed_num != 0:
        set_seed(int(seed_num))

    print(f"Generating audio for text: '{text}' on device: {DEVICE}")
    wav = model_obj.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        pace=pace,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    print("Audio generation complete.")
    # The model state is passed back out, which is correct for gr.State
    return (model_obj, (model_obj.sr, wav.squeeze(0).numpy()))


with gr.Blocks() as demo:
    # To ensure model loads on app start and uses DEVICE correctly:
    # Pre-load the model here if you want it loaded once globally for the Space instance.
    # However, with gr.State(None) and loading in `generate`,
    # the first user hitting "Generate" will trigger the load.
    # This is fine if `ChatterboxTTS.from_pretrained(DEVICE)` correctly uses the GPU
    # within the @spaces.GPU decorated `generate` function.

    # For better clarity on model loading with ZeroGPU:
    # Consider a dedicated function for loading the model that's called to initialize gr.State,
    # or ensure the first call to `generate` handles it robustly within the GPU context.
    # The current approach of loading if model_state is None within `generate` is okay
    # as long as `generate` itself is decorated.

    model_state = gr.State(None)

    with gr.Row():
        # ... (rest of your UI code is fine) ...
        with gr.Column():
            text = gr.Textbox(value="What does the fox say?", label="Text to synthesize")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/wav7604828.wav")
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
            model_state,
            text,
            ref_wav,
            exaggeration,
            pace,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[model_state, audio_output],
    )

# The share=True in launch() will give a UserWarning on Spaces, it's not needed.
# Hugging Face Spaces provides the public link automatically.
demo.queue(
        max_size=50,
        default_concurrency_limit=1, # Good for single model instance on GPU
    ).launch() # Removed share=True