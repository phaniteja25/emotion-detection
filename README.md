# Emotion Detection App

Streamlit app that predicts emotions (joy, sadness, anger, fear, surprise) from text using a fine-tuned XLM-R model.

## Demo 
https://github.com/user-attachments/assets/6c17e7da-589a-4010-a732-33e86aa692c0

## Prerequisites

- Python 3.9+ (3.10 or 3.11 recommended)
- Git

- Git LFS (required for the model weights)

## Setup (first time only)

1) Clone the repo
```
git clone <your-repo-url>
cd emotion-detection
```

2) Install Git LFS and fetch the model weights
```
git lfs install
git lfs pull
```

3) Create and activate a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```
.venv\Scripts\Activate.ps1
```

4) Install Python dependencies
```
pip install -r requirements.txt
```

## Run the app

```
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Troubleshooting

- Model load fails or `AttributeError: 'NoneType' object has no attribute 'to'`:
  - Ensure Git LFS is installed and the real model file is present.
  - The file `finetuned_model/model.safetensors` should be large (~1.1 GB), not a tiny text pointer.
  - Re-run:
    ```
    git lfs pull
    ```
- If you updated dependencies, reinstall them inside the active venv:
  ```
  pip install -r requirements.txt
  ```

## Project structure

- `app.py` - Streamlit app
- `finetuned_model/` - Fine-tuned model + tokenizer files
- `requirements.txt` - Python dependencies
