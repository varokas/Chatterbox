---
title: Chatterbox TTS
emoji: 🍿
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
short_description: Expressive Zeroshot TTS
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

* Needs Python>=3.10,<3.13 due to aifc missing

Source - git clone https://huggingface.co/spaces/ResembleAI/Chatterbox

## Local Setup Instructions

To run this project locally, you'll need to install `uv`. You can do this by running the following command in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once `uv` is installed, follow these steps:

1.  Create a virtual environment with Python 3.10 and activate it:
    ```bash
uv venv --python 3.10
source .venv/bin/activate
    ```
2.  Install the project dependencies:
    ```bash
uv pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
uv run app.py
    ```

Once the application is running, you can browse to `http://localhost:7860` in your web browser.
