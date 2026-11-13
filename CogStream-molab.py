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


@app.cell
def _(mo):
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    mo.output.clear_console()
    return (fs,)


@app.cell
def _(fs):
    # List only folders using detail=True (fewer API calls)
    all_items = fs.ls("Thesis-Data/CogStream-testing", detail=True)

    # Items are returned as a list of info dictionaries
    folders_only = [info["name"] for info in all_items if info.get("type") == "directory"]
    folders_only
    return


@app.cell
def _(fs):
    print(type(fs), fs.protocol)
    return


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
    # STEP 1: Ensure git-lfs and ffmpeg are Installed
    # ==============================================================================
    print("Installing system dependencies (git-lfs, ffmpeg)...", flush=True)
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq git-lfs ffmpeg", 
        shell=True, 
        check=True
    )
    subprocess.run(["git", "lfs", "install"], check=True)
    print("System dependencies ready.\n", flush=True)

    # ==============================================================================
    # STEP 2: Clone Code Repository
    # ==============================================================================
    REPO_URL = "https://github.com/chillCode404/CogStream-testing.git"
    ROOT_DIR = Path("./CogStream-testing").resolve()
    DATASET_DIR = ROOT_DIR / "Dataset"
    MODEL_DIR = ROOT_DIR / "model"

    if not ROOT_DIR.exists():
        print("Cloning CogStream code repository...", flush=True)
        subprocess.run(["git", "clone", REPO_URL, str(ROOT_DIR)], check=True)
        print("Code repository ready.\n", flush=True)
    else:
        print("Code repository already exists. Ensuring all tracked files are checked out...", flush=True)
        subprocess.run(
        ["git", "checkout", "HEAD", "--", "model/"],
        cwd=ROOT_DIR,
        check=True
    )


    # ==============================================================================
    # STEP 3: Sync Dataset via Git LFS (Safe Resume)
    # ==============================================================================
    print(f"Syncing dataset into {DATASET_DIR}...", flush=True)

    dataset_auth_url = f"https://user:{HF_TOKEN}@huggingface.co/datasets/SII-KYW/CogStream"

    dataset_sync_script = f"""
    set -e
    git lfs install
    git config --global lfs.concurrenttransfers 16
    git config --global http.postBuffer 1048576000

    # Case 1: First-time session (folder does not exist)
    if [ ! -d "{DATASET_DIR}" ]; then
        echo "Cloning dataset metadata and repository structure..."
        GIT_LFS_SKIP_SMUDGE=1 git clone {dataset_auth_url} {DATASET_DIR}
        cd {DATASET_DIR}

    # Case 2: Folder exists from snapshot_download, but has no .git folder
    elif [ ! -d "{DATASET_DIR}/.git" ]; then
        echo "Existing partial download detected. Initializing Git to resume..."
        cd {DATASET_DIR}
        git init
        git remote add origin {dataset_auth_url}
        git fetch origin main

    # Case 3: Already a Git repository
    else
        echo "Existing repository detected. Fetching latest remote commits..."
        cd {DATASET_DIR}
        git fetch origin main || true
    fi

    # ALWAYS guarantee all tracked JSON files and metadata are restored cleanly on disk
    echo "Restoring and verifying all tracked JSON and metadata files..."
    git reset --hard origin/main

    echo "Pulling video binary files (Git LFS) with 16 parallel streams..."
    git lfs pull
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
            allow_patterns=["*.safetensors"],
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
def _(ROOT_DIR, clean_env, subprocess):
    subprocess.run(
        ["git", "pull", "origin", "main"],
        cwd=str(ROOT_DIR),
        env=clean_env,
        check=True
    )
    return


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
    # allow pytorch expandable segments to reclaim fragmented gpu mem.
    clean_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    clean_env["PYTHONUNBUFFERED"] = "1"

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
    # STEP 2: Install PyTorch (CUDA 12.8 for Blackwell sm_120)
    # ==============================================================================
    print("Installing PyTorch (CUDA 12.8 for Blackwell sm_120)...", flush=True)
    subprocess.run(
        [
            "uv", "pip", "install", 
            "--python", str(ENV_PATH),
            "--extra-index-url", "https://download.pytorch.org/whl/cu128",
        "--index-strategy", "unsafe-best-match",
        "torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"
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
        "ffmpeg-python==0.2.0",
        "imageio-ffmpeg==0.4.9",
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
    # STEP 4: Attention Mechanism Setup (Native SDPA for Blackwell sm_120)
    # ==============================================================================
    print("Using PyTorch native SDPA (Scaled Dot-Product Attention) for Blackwell sm_120.\n", flush=True)

    # ==============================================================================
    # STEP 5: Verification (Run with clean_env)
    # ==============================================================================
    verify_cmd = (
        "import torch, transformers, peft, decord, huggingface_hub, filelock, ffmpeg; "
        "print('=' * 60); "
        "print(f'Python Executable: {torch.__file__}'); "
        "print(f'HuggingFace Hub:  {huggingface_hub.__file__}'); "
        "print(f'FileLock:         {filelock.__file__}'); "
        "print(f'PyTorch:          {torch.__version__} | CUDA: {torch.cuda.is_available()}'); "
        "print(f'CUDA Version:     {torch.version.cuda}'); "
        "print(f'GPU Device:       {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); "
        "print(f'Transformers:     {transformers.__version__}'); "
        "print(f'PEFT:             {peft.__version__}'); "
        "print(f'Decord:           {decord.__version__}'); "
        "print(f'FFmpeg Module:    {ffmpeg.__file__}'); "
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
        attn_implementation="sdpa",
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
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda:0")

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
    ## Testing with Zero-shot Model
    - Note: current code does not support running without LoRA modules yet.
    - Need at least the LoRA module for historic dialogue retrieval.
    - TODO: train HDR module (for 1-2 epoch for example usage), then find way to run without LoRA for video-text interleaved reasoning.
    """)
    return


@app.cell
def _(ROOT_DIR, TORCHRUN_BIN, clean_env, subprocess):
    # Launch zero-shot baseline evaluation
    zero_shot_cmd = [
        TORCHRUN_BIN,
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
    ## Stage 1: Historic Dialogue Retrieval
    """)
    return


@app.cell
def _(ROOT_DIR, TORCHRUN_BIN, clean_env, fs, subprocess):
    import hashlib as _hashlib
    import json as _json
    import re as _re
    from collections import deque
    from pathlib import Path as _Path
    from queue import Empty, SimpleQueue
    from tempfile import TemporaryDirectory as _TemporaryDirectory
    from threading import Event, Thread, enumerate as _enumerate_threads
    from time import monotonic
    from uuid import uuid4 as _uuid4

    # A timed-out wait does not kill an in-flight HTTP request. Refuse a rerun
    # while the previous worker is still alive, before touching fs or the log.
    if any(thread.name == "stage1-drive-sync" for thread in _enumerate_threads()):
        raise RuntimeError(
            "The previous Stage 1 Drive worker is still finishing a request. "
            "Wait for it to finish before rerunning this cell to avoid competing uploads."
        )

    remote_dir = "Thesis-Data/CogStream-testing/HDR_checkpoints"
    local_ckpt_dir = ROOT_DIR / "stage1_lora_weights_custom"
    log_file = ROOT_DIR / "stage1_training.log"
    audit_file = ROOT_DIR / "qa_format_audit.json"

    # Drive requests may spend a long time retrying. Keep all uploads in one
    # worker so those requests cannot block supervision of the training process.
    # These intervals control backup frequency, status reporting, and how long
    # the cell waits for backups AFTER it has already reported training's exit.
    _sync_interval_seconds = 60
    _heartbeat_interval_seconds = 60
    _monitor_interval_seconds = 1
    _final_sync_wait_seconds = 30
    _training_finished = Event()
    _cancel_sync = Event()
    _sync_busy = Event()
    _sync_messages = SimpleQueue()
    _manifest_name = "_checkpoint_manifest.json"

    def _checkpoint_step(name):
        if not _re.fullmatch(r"checkpoint-\d+", name):
            raise ValueError(f"Invalid checkpoint directory name: {name}")
        return int(name.split("-")[-1])

    def _check_sync_cancelled():
        if _cancel_sync.is_set():
            raise RuntimeError("Drive sync was cancelled")

    def _check_required_files(files):
        # These names match this launcher's single-process, non-DeepSpeed,
        # bf16 LoRA Trainer configuration (transformers 4.46.3). Adapter weights
        # alone are insufficient to resume the optimizer, LR schedule, and RNG.
        required = {
            "adapter_config.json", "trainer_state.json", "training_args.bin",
            "optimizer.pt", "scheduler.pt", "rng_state.pth",
        }
        missing = required - files.keys()
        if missing:
            raise ValueError(f"Missing resume files: {', '.join(sorted(missing))}")
        if not {"adapter_model.safetensors", "adapter_model.bin"}.intersection(files):
            raise ValueError("Missing LoRA adapter weights")
        for name, details in files.items():
            # Manifests must describe flat checkpoint files, never paths outside
            # the staging directory or the manifest itself.
            if not isinstance(name, str) or name in {"", ".", "..", _manifest_name} or "/" in name or "\\" in name:
                raise ValueError(f"Invalid checkpoint filename: {name!r}")
            if not isinstance(details, dict) or type(details.get("size")) is not int or details["size"] <= 0:
                raise ValueError(f"Missing/empty file or invalid size: {name}")

    def _check_manifest(manifest, checkpoint_name):
        if not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("complete") is not True:
            raise ValueError("Checkpoint manifest is invalid or upload is incomplete")
        if manifest.get("checkpoint") != checkpoint_name or type(manifest.get("global_step")) is not int:
            raise ValueError("Checkpoint manifest identifies a different checkpoint")
        if manifest["global_step"] != _checkpoint_step(checkpoint_name):
            raise ValueError("Checkpoint manifest step does not match its directory")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("Checkpoint manifest has no file inventory")
        _check_required_files(files)
        for name, details in files.items():
            if not isinstance(details.get("sha256"), str) or not _re.fullmatch(r"[0-9a-f]{64}", details["sha256"]):
                raise ValueError(f"Invalid SHA-256 checksum for {name}")

    def _checkpoint_snapshot(directory):
        step = _checkpoint_step(directory.name)
        files = {}
        for path in sorted(directory.iterdir()):
            if path.name == _manifest_name:
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Unsupported checkpoint entry: {path.name}")
            files[path.name] = {"size": path.stat().st_size}
        _check_required_files(files)
        state = _json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
        config = _json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        if not isinstance(state, dict) or type(state.get("global_step")) is not int or state["global_step"] != step:
            raise ValueError("trainer_state.json step does not match its directory")
        if not isinstance(config, dict) or not config:
            raise ValueError("adapter_config.json must contain a configuration object")
        for name, details in files.items():
            # Stream hashes in small blocks: optimizer checkpoints can be large.
            # Do not load torch/pickle files merely to check transport integrity.
            digest = _hashlib.sha256()
            with open(directory / name, "rb") as source:
                while block := source.read(1024 * 1024):
                    _check_sync_cancelled()
                    digest.update(block)
            details["sha256"] = digest.hexdigest()
        return {"version": 1, "complete": True, "checkpoint": directory.name, "global_step": step, "files": files}

    def _validate_local_checkpoint(directory):
        snapshot = _checkpoint_snapshot(directory)
        marker = directory / _manifest_name
        if marker.exists():
            manifest = _json.loads(marker.read_text(encoding="utf-8"))
            _check_manifest(manifest, directory.name)
            if manifest["files"] != snapshot["files"]:
                raise ValueError("Local checkpoint size/SHA-256 mismatch")
        return snapshot

    def _remote_files(directory):
        # Refresh listings after writes; cached sizes cannot verify an upload.
        fs.invalidate_cache(directory)
        files = {}
        for info in fs.ls(directory, detail=True):
            if info.get("type") != "directory":
                name = info["name"].rstrip("/").split("/")[-1]
                if name in files:
                    raise ValueError(f"Ambiguous duplicate Drive filename: {name}")
                files[name] = info
        return files

    def _drain_sync_messages():
        # Print from the cell's thread so worker messages reach notebook output.
        while True:
            try:
                print(_sync_messages.get_nowait(), flush=True)
            except Empty:
                break

    def _upload_file(local_path, remote_path):
        if not _cancel_sync.is_set() and local_path.is_file():
            try:
                fs.put(str(local_path), remote_path)
                return True
            except Exception as e:
                _sync_messages.put(f"[Drive Sync Warning] Failed to upload {local_path.name}: {e}")
        return False

    def _upload_checkpoint(local_dir, remote_target):
        if _cancel_sync.is_set():
            return False
        try:
            manifest = _validate_local_checkpoint(local_dir)
            fs.makedirs(remote_target, exist_ok=True)
            with _TemporaryDirectory(prefix="cogstream-manifest-") as temporary:
                marker = _Path(temporary) / _manifest_name
                # Invalidate any previous completion marker BEFORE overwriting
                # payloads. This also distinguishes interrupted new uploads from
                # older, pre-manifest backups that may still be recoverable.
                marker.write_text(_json.dumps({**manifest, "complete": False}), encoding="utf-8")
                _check_sync_cancelled()
                fs.put(str(marker), f"{remote_target}/{_manifest_name}")
                for name in manifest["files"]:
                    _check_sync_cancelled()
                    fs.put(str(local_dir / name), f"{remote_target}/{name}")
                _check_sync_cancelled()
                remote_files = _remote_files(remote_target)
                for name, details in manifest["files"].items():
                    if remote_files.get(name, {}).get("size") != details["size"]:
                        raise ValueError(f"Uploaded file size mismatch: {name}")
                # Trainer may rotate/delete a checkpoint during a slow upload.
                # Publish completion only if its local contents are unchanged.
                if _checkpoint_snapshot(local_dir) != manifest:
                    raise ValueError("Local checkpoint changed during upload")
                marker.write_text(_json.dumps(manifest, sort_keys=True), encoding="utf-8")
                _check_sync_cancelled()
                fs.put(str(marker), f"{remote_target}/{_manifest_name}")
                _check_sync_cancelled()
                verified_marker = _Path(temporary) / "verified-manifest.json"
                fs.get(f"{remote_target}/{_manifest_name}", str(verified_marker))
                if _json.loads(verified_marker.read_text(encoding="utf-8")) != manifest:
                    raise ValueError("Uploaded completion manifest failed read-back validation")
            _sync_messages.put(f"[Drive Sync] Validated backup: {local_dir.name}.")
            return True
        except Exception as e:
            _sync_messages.put(f"[Drive Sync Warning] Backup of {local_dir.name} was not confirmed; will retry on a later pass: {e}")
            return False

    # Trainer chooses the highest checkpoint-* directory without validating it.
    # Move invalid local directories out of that namespace, preserving their
    # contents for inspection rather than deleting potentially useful weights.
    uploaded_ckpts = set()
    local_ckpts = []
    # Remember rejected local checkpoints across cell reruns too. Moving a bad
    # directory aside must not make the next run silently start from scratch.
    _quarantine_dir = local_ckpt_dir / ".incomplete"
    _had_invalid_checkpoint = _quarantine_dir.is_dir() and any(_quarantine_dir.iterdir())
    for candidate in sorted(local_ckpt_dir.glob("checkpoint-*")):
        if not candidate.is_dir():
            continue
        try:
            _validate_local_checkpoint(candidate)
            local_ckpts.append(candidate)
        except Exception as e:
            _had_invalid_checkpoint = True
            _quarantine_dir.mkdir(exist_ok=True)
            preserved = _quarantine_dir / f"{candidate.name}-{_uuid4().hex}"
            candidate.rename(preserved)
            print(f"[Checkpoint Warning] Preserved invalid {candidate.name} at {preserved}: {e}", flush=True)

    # Restore only the newest VALID remote checkpoint. Download into a hidden
    # staging directory on the same filesystem, then rename it atomically so a
    # failed/interrupted download cannot become Trainer's next resume candidate.
    if not local_ckpts and fs.exists(remote_dir):
        print(f"Checking {remote_dir} for validated checkpoints to resume...", flush=True)
        candidates = [
            info for info in fs.ls(remote_dir, detail=True)
            if info.get("type") == "directory"
            and _re.fullmatch(r"checkpoint-\d+", info["name"].rstrip("/").split("/")[-1])
        ]
        candidates.sort(key=lambda info: _checkpoint_step(info["name"].rstrip("/").split("/")[-1]), reverse=True)
        for info in candidates:
            name = info["name"].rstrip("/").split("/")[-1]
            try:
                remote_files = _remote_files(info["name"])
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                with _TemporaryDirectory(dir=local_ckpt_dir, prefix=".restore-") as temporary:
                    staged = _Path(temporary) / name
                    staged.mkdir()
                    manifest = None
                    if _manifest_name in remote_files:
                        fs.get(f"{info['name']}/{_manifest_name}", str(staged / _manifest_name))
                        manifest = _json.loads((staged / _manifest_name).read_text(encoding="utf-8"))
                        _check_manifest(manifest, name)
                        expected_files = manifest["files"]
                    else:
                        # Existing user backups have no manifest. Require the
                        # full resume file set and advertised sizes; historical
                        # corruption cannot be ruled out without prior hashes.
                        expected_files = remote_files
                        _check_required_files(expected_files)
                        print(f"[Checkpoint Warning] Legacy {name}: validating required files and sizes; no prior hashes available.", flush=True)
                    for fname, details in expected_files.items():
                        if remote_files.get(fname, {}).get("size") != details["size"]:
                            raise ValueError(f"Remote file missing or size mismatch: {fname}")
                        fs.get(f"{info['name']}/{fname}", str(staged / fname))
                        if (staged / fname).stat().st_size != details["size"]:
                            raise ValueError(f"Downloaded file size mismatch: {fname}")
                    _validate_local_checkpoint(staged)
                    restored = local_ckpt_dir / name
                    staged.rename(restored)
                local_ckpts.append(restored)
                if manifest is not None:
                    # Only a backup actually restored and verified this run is
                    # known to exist remotely. Other local checkpoints must sync.
                    uploaded_ckpts.add(name)
                print(f"[Checkpoint] Restored validated {name}.", flush=True)
                break
            except Exception as e:
                _had_invalid_checkpoint = True
                print(f"[Checkpoint Warning] Skipping {name}: {e}", flush=True)
    if not local_ckpts and _had_invalid_checkpoint:
        raise RuntimeError(
            "Existing checkpoints failed validation; training was not started from scratch. "
            "See the checkpoint warnings above; rejected local files, if any, are preserved in .incomplete."
        )

    stage1_cmd = [
        TORCHRUN_BIN,
        "--nproc_per_node=1",
        "train/first_stage_training.py",
        "--model_path", "./model",
        "--QA_path", "./Dataset/Train_dataset/QA_Dataset",
        "--resume_from_checkpoint", "./stage1_lora_weights_custom"
    ]

    print("Starting Stage 1 training (Historic Dialogue Retrieval)...", flush=True)
    print(f"All output is being written to: {log_file}", flush=True)
    print(f"Checkpoints & logs will auto-sync to Google Drive: {remote_dir}", flush=True)
    print(f"Monitor in real time from the Molab terminal with:\n    tail -f {log_file}", flush=True)

    def _drive_sync_worker():
        # This worker alone owns upload bookkeeping and uses fs after restore.
        # Never start a second uploader if this one stalls: that would pile up
        # requests and permit concurrent writes to the same remote files.
        audit_uploaded = False
        while not _cancel_sync.is_set():
            # Setting this event wakes the worker for a final pass immediately,
            # rather than waiting out the normal 60-second backup interval.
            final_pass = _training_finished.wait(_sync_interval_seconds)
            if _cancel_sync.is_set():
                return
            _sync_busy.set()
            try:
                # trainer_state.json is an initial readiness hint; the upload
                # helper validates all required resume files before publishing.
                if local_ckpt_dir.exists():
                    for ckpt in sorted(local_ckpt_dir.glob("checkpoint-*")):
                        if _cancel_sync.is_set():
                            return
                        if ckpt.name not in uploaded_ckpts and (ckpt / "trainer_state.json").is_file():
                            _sync_messages.put(f"[Drive Sync] Uploading {ckpt.name}...")
                            if _upload_checkpoint(ckpt, f"{remote_dir}/{ckpt.name}"):
                                uploaded_ckpts.add(ckpt.name)
                    if final_pass:
                        for item in local_ckpt_dir.iterdir():
                            _upload_file(item, f"{remote_dir}/{item.name}")

                _upload_file(log_file, f"{remote_dir}/stage1_training.log")
                if not audit_uploaded and audit_file.is_file():
                    audit_uploaded = _upload_file(audit_file, f"{remote_dir}/qa_format_audit.json")
            except Exception as e:
                # For example, Trainer may rotate a checkpoint while we scan
                # local files. A backup error must not terminate monitoring.
                _sync_messages.put(f"[Drive Sync Warning] Backup pass failed: {e}")
            finally:
                _sync_busy.clear()
            if final_pass:
                return

    # A daemon worker cannot keep the notebook kernel alive during shutdown.
    # Python cannot forcibly stop a thread inside fs.put(); cancellation below
    # prevents subsequent uploads once that outstanding request returns.
    _sync_thread = Thread(target=_drive_sync_worker, name="stage1-drive-sync", daemon=True)
    try:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                stage1_cmd,
                cwd=str(ROOT_DIR),
                # Flush Python output promptly into the local log, including
                # output from workers launched by torchrun. Leave clean_env intact.
                env={**clean_env, "PYTHONUNBUFFERED": "1"},
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            _sync_thread.start()
            _next_heartbeat = monotonic() + _heartbeat_interval_seconds
            while True:
                try:
                    proc.wait(timeout=_monitor_interval_seconds)
                    break
                except subprocess.TimeoutExpired:
                    pass
                _drain_sync_messages()
                if monotonic() >= _next_heartbeat:
                    # Process liveness is not proof of optimizer progress. The
                    # local log size helps distinguish it from a stale Drive copy.
                    _sync_status = "busy" if _sync_busy.is_set() else "waiting"
                    print(
                        f"[Stage 1] Training process {proc.pid} is still running; "
                        f"local log: {log_file.stat().st_size:,} bytes; "
                        f"Drive sync: {_sync_status}.",
                        flush=True,
                    )
                    _next_heartbeat = monotonic() + _heartbeat_interval_seconds

        # Report the LOCAL result before requesting or waiting for final backup.
        # Even if Drive is stuck, a CUDA traceback/torchrun failure is visible now.
        print(f"\n[Stage 1] Training process exited with code {proc.returncode}.", flush=True)
        print("\n--- Last 50 lines of local training log ---", flush=True)
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in deque(f, maxlen=50):
                print(line, end="", flush=True)
        _drain_sync_messages()

        _training_finished.set()
        print(f"\n[Drive Sync] Waiting up to {_final_sync_wait_seconds}s for final backup...", flush=True)
        _sync_thread.join(timeout=_final_sync_wait_seconds)
        if _sync_thread.is_alive():
            _cancel_sync.set()
            print(
                "[Drive Sync Warning] Final backup wait expired; remote files may be stale or incomplete. "
                "An in-flight Drive request may still finish, but no further uploads will start. "
                f"Local training output remains at {ROOT_DIR}.",
                flush=True,
            )
        _drain_sync_messages()

        # Backup timing/errors cannot replace the training process's exit code.
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, stage1_cmd)
        print("[Stage 1] Training completed successfully.", flush=True)
    finally:
        # Also wake/cancel an idle backup worker if the cell is interrupted.
        _cancel_sync.set()
        _training_finished.set()
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
        "--config_file", "default_config.yaml",
        "--num_processes=1",  # Overrides 'num_processes: 8' for your single Blackwell GPU
        "--num_epochs=1", # train for 1 epoch to test the results first.
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
