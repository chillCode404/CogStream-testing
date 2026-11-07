# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "hf-transfer==0.1.9",
#     "huggingface-hub==2.2.0",
#     "python-dotenv==1.2.4",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import subprocess
    from pathlib import Path

    return Path, os, subprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Download code repo
    """)
    return


@app.cell
def _(Path, os, subprocess):
    # ==============================================================================
    # STEP 0: Load Secrets from .env
    # ==============================================================================
    try:
        from dotenv import load_dotenv
    except ImportError:
        subprocess.run(["pip", "install", "-q", "--no-cache-dir", "python-dotenv"], check=True)
        from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=True)
    HF_TOKEN = os.getenv("HF_TOKEN")

    if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
        raise ValueError("HF_TOKEN was not found or is invalid in .env")

    # ==============================================================================
    # STEP 1: Ensure git-lfs is Installed
    # ==============================================================================
    check_lfs = subprocess.run(["which", "git-lfs"], capture_output=True, text=True)
    if check_lfs.returncode != 0:
        print("git-lfs not detected. Installing via apt-get...", flush=True)
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq git-lfs", 
            shell=True, 
            check=True
        )
        subprocess.run(["git", "lfs", "install"], check=True)
        print("git-lfs installed successfully.\n", flush=True)

    # ==============================================================================
    # STEP 2: Clone Code Repository
    # ==============================================================================
    REPO_URL = "https://github.com/chillCode404/CogStream-testing.git"
    ROOT_DIR = Path("./CogStream-testing")
    DATASET_DIR = ROOT_DIR / "Dataset"
    MODEL_DIR = ROOT_DIR / "model"

    if not ROOT_DIR.exists():
        print("Cloning CogStream code repository...", flush=True)
        subprocess.run(["git", "clone", REPO_URL, str(ROOT_DIR)], check=True)
        print("Code repository ready.\n", flush=True)
    else:
        print("Code repository already exists. Skipping clone.\n", flush=True)

    # ==============================================================================
    # STEP 3: Sync Dataset via Git LFS (Safe Resume)
    # ==============================================================================
    print(f"Syncing dataset into {DATASET_DIR}...", flush=True)

    dataset_auth_url = f"https://user:{HF_TOKEN}@huggingface.co/datasets/SII-KYW/CogStream"

    dataset_sync_script = f"""
    set -e
    git lfs install

    # Case 1: First-time session (folder does not exist)
    if [ ! -d "{DATASET_DIR}" ]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone {dataset_auth_url} {DATASET_DIR}
        cd {DATASET_DIR}
        git lfs pull

    # Case 2: Folder exists from snapshot_download, but has no .git folder
    elif [ ! -d "{DATASET_DIR}/.git" ]; then
        echo "Existing partial download detected. Initializing Git to resume..."
        cd {DATASET_DIR}
        git init
        git remote add origin {dataset_auth_url}
        git fetch origin main
        git reset origin/main
        git lfs pull

    # Case 3: Already a Git repository
    else
        cd {DATASET_DIR}
        git lfs pull
    fi
    """

    subprocess.run(dataset_sync_script, shell=True, executable="/bin/bash", check=True)
    print("CogStream dataset fully synced and verified.\n", flush=True)

    # ==============================================================================
    # STEP 4: Download VideoLLaMA3-7B Weights (Safetensors Check)
    # ==============================================================================
    # The GitHub repo already includes the JSON configs, so we MUST check for 
    # the actual ~14 GB binary weight shards (*.safetensors).
    safetensor_shards = list(MODEL_DIR.glob("*.safetensors"))

    # VideoLLaMA3-7B has 4 safetensor shards
    if len(safetensor_shards) < 4:
        print(f"Weight shards missing (found {len(safetensor_shards)}/4). Downloading VideoLLaMA3-7B weights (~14 GB)...", flush=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="DAMO-NLP-SG/VideoLLaMA3-7B",
            repo_type="model",
            local_dir=str(MODEL_DIR),
            max_workers=4,
            token=HF_TOKEN
        )
        print("Model weight shards successfully downloaded.\n", flush=True)
    else:
        print(f"All {len(safetensor_shards)} model weight shards (*.safetensors) are present. Skipping download.\n", flush=True)

    # ==============================================================================
    # STEP 5: Verification
    # ==============================================================================
    print("=" * 45)
    print("SETUP COMPLETE - VERIFYING DIRECTORIES:")
    print("=" * 45)
    print(f"Dataset root contains: {os.listdir(DATASET_DIR)}")
    if (DATASET_DIR / "Test_dataset" / "event_segments").exists():
        num_test = len(os.listdir(DATASET_DIR / "Test_dataset" / "event_segments"))
        print(f"Test split: {num_test} video directories verified")
    if (DATASET_DIR / "Train_dataset" / "event_segments").exists():
        num_train = len(os.listdir(DATASET_DIR / "Train_dataset" / "event_segments"))
        print(f"Train split: {num_train} video directories verified")
    print(f"Model directory: {len(os.listdir(MODEL_DIR))} files verified")
    return DATASET_DIR, ROOT_DIR


@app.cell
def _(DATASET_DIR, subprocess):
    # DATASET_DIR = Path("./CogStream-testing/Dataset").resolve()

    checkout_script = f"""
    set -e
    cd {DATASET_DIR}

    echo "=== Current Git Status ==="
    git status --short | head -n 20

    echo -e "\nRestoring all tracked JSON and metadata files to disk..."
    git reset --hard origin/main

    echo -e "\n=== Directory Contents After Restore ==="
    ls -la
    """

    result = subprocess.run(checkout_script, shell=True, executable="/bin/bash", capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Stderr:", result.stderr)
    return


@app.cell
def _():
    # result = subprocess.run(
    #     ["ls", "-la", "./CogStream-testing/model/"],
    #     capture_output=True,
    #     text=True,
    #     check=True
    # )
    # print(result.stdout)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Python Env Setup
    """)
    return


@app.cell
def _(Path, os, subprocess):
    ENV_PATH = Path("/tmp/cogstream_env")
    PYTHON_BIN = ENV_PATH / "bin" / "python"

    # ==============================================================================
    # CREATE CLEAN SUBPROCESS ENVIRONMENT (ISOLATION)
    # ==============================================================================
    # Remove host Python 3.13 paths so they cannot leak into Python 3.10
    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env.pop("PYTHONHOME", None)
    clean_env["VIRTUAL_ENV"] = str(ENV_PATH)
    clean_env["PATH"] = f"{ENV_PATH / 'bin'}:{clean_env.get('PATH', '')}"

    # ==============================================================================
    # STEP 1: Create Isolated Python 3.10 Virtual Environment
    # ==============================================================================
    if not ENV_PATH.exists():
        print("Creating clean Python 3.10 environment at /tmp/cogstream_env...", flush=True)
        subprocess.run(
            ["uv", "venv", "--python", "3.10", str(ENV_PATH)],
            env=clean_env,
            check=True
        )
        print("Environment created.\n", flush=True)
    else:
        print(f"Environment exists at {ENV_PATH}.\n", flush=True)

    # ==============================================================================
    # STEP 2: Install PyTorch 2.4.0 (CUDA 12.4)
    # ==============================================================================
    print("Installing PyTorch 2.4.0 (CUDA 12.4)...", flush=True)
    subprocess.run(
        [
            "uv", "pip", "install", 
            "--python", str(ENV_PATH),
            "--extra-index-url", "https://download.pytorch.org/whl/cu124",
            "--index-strategy", "unsafe-best-match",
            "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0"
        ],
        env=clean_env,
        check=True
    )
    print("PyTorch installed.\n", flush=True)

    # ==============================================================================
    # STEP 3: Install Pinned CogStream & VideoLLaMA3 Dependencies
    # ==============================================================================
    packages = [
        "huggingface_hub",
        "filelock",
        "transformers==4.46.3",
        "peft==0.15.2",
        "accelerate==1.0.1",
        "deepspeed==0.15.4",
        "decord==0.6.0",
        "timm==1.0.3",
        "einops==0.6.1",
        "moviepy==1.0.3",
        "opencv-python",
        "av",
        "pandas",
        "scikit-learn",
        "sentencepiece"
    ]

    print("Installing CogStream & VideoLLaMA3 dependencies into venv...", flush=True)
    subprocess.run(
        [
            "uv", "pip", "install", 
            "--python", str(ENV_PATH),
            "--index-strategy", "unsafe-best-match",
            *packages
        ],
        env=clean_env,
        check=True
    )
    print("Dependencies installed.\n", flush=True)

    # ==============================================================================
    # STEP 4: Install Pre-Built Flash-Attention Wheel
    # ==============================================================================
    flash_attn_wheel = (
        "https://github.com/Dao-AILab/flash-attention/releases/download/"
        "v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )

    print("Installing pre-built flash-attn wheel...", flush=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(ENV_PATH), flash_attn_wheel],
        env=clean_env,
        check=True
    )
    print("FlashAttention installed.\n", flush=True)

    # ==============================================================================
    # STEP 5: Verification (Run with clean_env)
    # ==============================================================================
    verify_cmd = (
        "import torch, flash_attn, transformers, peft, decord, huggingface_hub, filelock; "
        "print('=' * 60); "
        "print(f'Python Executable: {torch.__file__}'); "
        "print(f'HuggingFace Hub:  {huggingface_hub.__file__}'); "
        "print(f'FileLock:         {filelock.__file__}'); "
        "print(f'PyTorch:          {torch.__version__} | CUDA: {torch.cuda.is_available()}'); "
        "print(f'GPU Device:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); "
        "print(f'Flash Attention:  {flash_attn.__version__}'); "
        "print(f'Transformers:     {transformers.__version__}'); "
        "print(f'PEFT:             {peft.__version__}'); "
        "print(f'Decord:           {decord.__version__}'); "
        "print('=' * 60); "
        "print('ALL CORE MODULES VERIFIED CLEANLY IN PYTHON 3.10!')"
    )

    subprocess.run(
        [str(PYTHON_BIN), "-c", verify_cmd],
        env=clean_env,
        check=True
    )
    return PYTHON_BIN, clean_env


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup environment variables
    """)
    return


@app.cell
def _(Path):
    # Paths to your isolated Python 3.10 environment and cloned repo
    ENV_BIN = Path("/tmp/cogstream_env/bin")

    TORCHRUN_BIN = str(ENV_BIN / "torchrun")
    ACCELERATE_BIN = str(ENV_BIN / "accelerate")
    return ACCELERATE_BIN, TORCHRUN_BIN


@app.cell
def _(Path):
    script_content = '''import os
    import json
    import torch
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoProcessor

    ROOT_DIR = Path("./CogStream-testing")
    MODEL_PATH = str(ROOT_DIR / "model")
    VIDEO_DIR = ROOT_DIR / "Dataset" / "Test_dataset" / "event_segments"
    QA_DIR = ROOT_DIR / "Dataset" / "Test_dataset" / "QA_Dataset"

    # 1. Locate test QA annotations
    qa_files = sorted(list(QA_DIR.glob("*.json")))
    if not qa_files:
        raise FileNotFoundError(f"No QA JSON files found in {QA_DIR}")

    sample_qa_file = qa_files[0]
    with open(sample_qa_file, "r") as f:
        sample_data = json.load(f)

    video_id = sample_qa_file.stem
    first_qa = sample_data[0] if isinstance(sample_data, list) else sample_data
    question_text = first_qa.get("question", first_qa.get("Q", ""))
    ground_truth = first_qa.get("answer", first_qa.get("A", ""))

    print("=" * 60)
    print(f"Testing on Video: {video_id}")
    print(f"Sample Question:  {question_text}")
    print(f"Ground Truth:     {ground_truth}")
    print("=" * 60)

    # 2. Load Base Model directly into GPU (Blackwell)
    print("\\nLoading VideoLLaMA3-7B base model into GPU...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0"
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Base model and processor loaded successfully.\\n", flush=True)

    # 3. Locate Video Clip
    video_folder = VIDEO_DIR / video_id
    video_files = list(video_folder.glob("*.mp4")) if video_folder.exists() else []

    if not video_files:
        direct_mp4 = VIDEO_DIR / f"{video_id}.mp4"
        if direct_mp4.exists():
            video_files = [direct_mp4]

    if video_files:
        sample_video_path = str(video_files[0])
        print(f"Processing video clip: {sample_video_path}")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample_video_path},
                    {"type": "text", "text": question_text}
                ]
            }
        ]
    else:
        print(f"Notice: Video clip for {video_id} not found in {VIDEO_DIR}, testing text path.")
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question_text}]
            }
        ]

    # 4. Generate Output
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, videos=[sample_video_path] if video_files else None, return_tensors="pt").to("cuda:0")

    print("\\nGenerating model response...", flush=True)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    input_len = inputs.input_ids.shape[1]
    generated_tokens = output_ids[0, input_len:]
    prediction = processor.decode(generated_tokens, skip_special_tokens=True).strip()

    print("=" * 60)
    print(f"Model Prediction: {prediction}")
    print(f"Ground Truth:     {ground_truth}")
    print("=" * 60)
    print("\\nPIPELINE VERIFIED: Base VideoLLaMA3 inference executed successfully!")
    '''

    Path("smoke_test.py").write_text(script_content)
    print("Saved smoke_test.py to workspace.")
    return


@app.cell
def _(PYTHON_BIN, clean_env, subprocess):
    subprocess.run(
        [PYTHON_BIN, "smoke_test.py"],
        env=clean_env,
        check=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Testing CogReasoner
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing with Zero-shot Model
    - Note: current code does not support running without LoRA modules yet.
    """)
    return


@app.cell
def _(PYTHON_BIN, ROOT_DIR, clean_env, subprocess):
    # Launch zero-shot baseline evaluation
    zero_shot_cmd = [
        PYTHON_BIN,
        "evaluate/answer_generate.py",
        "--model_path", "./model",
        "--video_dir", "./Dataset/Test_dataset/event_segments",
        "--query_dir", "./Dataset/Test_dataset/QA_Dataset",
        "--lora_adapter_1_path", "none",
        "--lora_adapter_2_path", "none"
    ]

    subprocess.run(zero_shot_cmd, cwd=str(ROOT_DIR), env=clean_env, check=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training + testing fine-tuned model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stage 1 (Historic Dialogue Retrieval) Training
    """)
    return


@app.cell
def _(ROOT_DIR, TORCHRUN_BIN, clean_env, subprocess):
    stage1_cmd = [
        TORCHRUN_BIN,
        "--nproc_per_node=1",
        "train/language_model_training.py",
        "--model_path", "./model",
        "--QA_path", "./Dataset/Train_dataset/QA_Dataset"
    ]

    print("Starting Stage 1 training (Historic Dialogue Retrieval)...", flush=True)

    subprocess.run(
        stage1_cmd,
        cwd=str(ROOT_DIR),    # Must be run from repo root for path resolution
        env=clean_env,        # Enforces Python 3.10 isolation
        check=True
    )

    print("Stage 1 training completed successfully.", flush=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Stage 2 (Video-text Interleaved Reasoning) Training
    """)
    return


@app.cell
def _(ACCELERATE_BIN, ROOT_DIR, clean_env, subprocess):
    stage2_cmd = [
        ACCELERATE_BIN,
        "launch",
        "--config_file", "default_configs.yaml",
        "--num_processes=1",  # Overrides 'num_processes: 8' for your single Blackwell GPU
        "--main_process_port=29501",
        "train/second_stage_training.py",
        "--model_path", "./model",
        "--video_dir", "./Dataset/Train_dataset/event_segments",
        "--query_dir", "./Dataset/Train_dataset/QA_Dataset"
    ]

    print("Starting Stage 2 training (Video-Text Interleave Reasoning)...", flush=True)

    subprocess.run(
        stage2_cmd,
        cwd=str(ROOT_DIR),
        env=clean_env,
        check=True
    )

    print("Stage 2 training completed successfully.", flush=True)
    return


if __name__ == "__main__":
    app.run()
