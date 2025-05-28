import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS # Assuming this path is correct
import gradio as gr
import spaces

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
# Load the model once when the application starts.
# This model will be accessible by the @spaces.GPU decorated function.
MODEL = None

def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Global MODEL is None, loading...")
        try:
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
            # Ensure model is on the correct device if not handled by from_pretrained
            if DEVICE == "cuda" and hasattr(MODEL, 'to'):
                MODEL.to(DEVICE)
            print(f"Global MODEL loaded. Device: {DEVICE}")
            if hasattr(MODEL, 'device'): # If the model object has a device attribute
                 print(f"Model internal device attribute: {MODEL.device}")
        except Exception as e:
            print(f"Error loading global model: {e}")
            raise
    return MODEL

# Attempt to load the model at startup.
# If this fails, the app will likely fail to start, which is informative.
try:
    get_or_load_model()
except Exception as e:
    # Handle critical model loading failure if necessary, or let it propagate
    print(f"CRITICAL: Failed to load model on startup. Error: {e}")
    # You might want to display an error in Gradio if this happens,
    # but for now, a print is fine for debugging.

def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

@spaces.GPU # Your GPU-accelerated function
def generate_tts_audio(text_input, audio_prompt_path_input, exaggeration_input, temperature_input, seed_num_input, cfgw_input):
    current_model = get_or_load_model() # Access the global model

    if current_model is None:
        # This should ideally not happen if startup loading was successful
        # Or, it indicates an issue with the global model pattern in this specific env.
        raise RuntimeError("Model could not be loaded or accessed.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(f"Generating audio for text: '{text_input}'")
    wav = current_model.generate(
        text_input,
        audio_prompt_path=audio_prompt_path_input,
        exaggeration=exaggeration_input,
        temperature=temperature_input,
        cfg_weight=cfgw_input,
    )
    print("Audio generation complete.")
    # ONLY return pickleable data
    return (current_model.sr, wav.squeeze(0).numpy())


with gr.Blocks() as demo:
    # No gr.State needed for the model object if it's managed globally
    # and not passed back and forth.

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(value="What does the fox say?", label="Text to synthesize")
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value="https://storage.googleapis.com/chatterbox-demo-samples/prompts/wav7604828.wav")
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)


            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    run_btn.click(
        fn=generate_tts_audio, # Use the new function name
        inputs=[
            # model_state, # Removed: model is now global
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output], # Only outputting the audio data
    )

demo.queue(
    max_size=50,
    default_concurrency_limit=1, # Important for a single global model
).launch() # share=True is not needed and causes a warning on Spaces