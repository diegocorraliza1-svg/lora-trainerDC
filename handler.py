import runpod
import subprocess, os, requests, zipfile, shutil, re, time

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dksemexxbmgmtdfbidnk.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def upload_to_supabase(local_path: str, storage_path: str, bucket: str = "loras", max_retries: int = 3):
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{storage_path}"
    with open(local_path, "rb") as f:
        data = f.read()
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true",
    }
    print(f"[upload] File size: {len(data) / 1024 / 1024:.1f} MB")
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, data=data, timeout=300)
            if r.ok:
                print(f"[upload] OK → {bucket}/{storage_path}")
                return
            print(f"[upload] Attempt {attempt}/{max_retries} failed {r.status_code}: {r.text}")
        except requests.exceptions.RequestException as e:
            print(f"[upload] Attempt {attempt}/{max_retries} network error: {e}")
        if attempt < max_retries:
            wait = 2 ** attempt
            print(f"[upload] Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to upload {storage_path} after {max_retries} attempts")


def handler(event):
    inp = event["input"]
    dataset_url = inp["dataset_zip_url"]
    trigger     = inp["trigger_word"]
    steps       = inp.get("steps", 2000)
    lr          = inp.get("learning_rate", 1e-4)
    batch       = inp.get("batch_size", 1)
    rank        = inp.get("rank", 32)
    alpha       = inp.get("alpha", 16)
    lora_name   = inp.get("lora_name", f"{trigger}_lora")
    project_id  = inp.get("project_id", "unknown")
    grad_ckpt   = inp.get("gradient_checkpointing", True)

    # Clean previous runs
    for d in ["/tmp/training", "/tmp/output"]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Download dataset
    zip_path = "/tmp/dataset.zip"
    print(f"[dataset] Downloading {dataset_url[:80]}...")
    r = requests.get(dataset_url, timeout=300)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)
    print(f"[dataset] Downloaded {len(r.content) / 1024 / 1024:.1f} MB")

    # Extract
    extract_dir = "/tmp/training/dataset"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)

    # Organize into Kohya format
    kohya_dir = f"/tmp/training/dataset/1_{trigger}"
    os.makedirs(kohya_dir, exist_ok=True)
    img_count = 0
    for root, _, files in os.walk(extract_dir):
        if root == kohya_dir:
            continue
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                shutil.move(os.path.join(root, fname), os.path.join(kohya_dir, fname))
                img_count += 1
    print(f"[dataset] {img_count} images organized into {kohya_dir}")

    # Find training script
    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    train_script = "/app/kohya/sd_scripts/sdxl_train_network.py"
    if not os.path.exists(train_script):
        train_script = "/app/kohya/sdxl_train_network.py"
    print(f"[train] Using script: {train_script}")

    cmd = [
        "accelerate", "launch", "--num_cpu_threads_per_process=1",
        train_script,
        "--pretrained_model_name_or_path=/models/sdxl/sd_xl_base_1.0.safetensors",
        f"--train_data_dir=/tmp/training/dataset",
        f"--output_dir={output_dir}",
        f"--output_name={lora_name}",
        "--network_module=networks.lora",
        f"--network_dim={rank}",
        f"--network_alpha={alpha}",
        f"--max_train_steps={steps}",
        f"--learning_rate={lr}",
        f"--train_batch_size={batch}",
        "--resolution=1024,1024",
        "--enable_bucket",
        "--mixed_precision=fp16",
        "--save_precision=fp16",
        "--optimizer_type=AdamW8bit",
        "--lr_scheduler=cosine",
        "--cache_latents",
        "--no_half_vae",
        "--sdpa",
        "--save_model_as=safetensors",
    ]

    if grad_ckpt:
        cmd.append("--gradient_checkpointing")

    logs = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        logs.append(line.rstrip())
        match = re.search(r'(\d+)/(\d+)', line)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            pct = min(99, int(current / total * 100))
            runpod.serverless.progress_update(event, {"percent": pct, "step": current, "total": total})
    proc.wait()

    if proc.returncode != 0:
        return {"error": "\n".join(logs[-50:])}

    safetensors = os.path.join(output_dir, f"{lora_name}.safetensors")
    if not os.path.exists(safetensors):
        return {"error": "Output not found. Logs:\n" + "\n".join(logs[-30:])}

    # Upload BEFORE reporting completed (prevents race condition)
    storage_path = f"{project_id}/{lora_name}.safetensors"
    upload_to_supabase(safetensors, storage_path)

    return {
        "status": "completed",
        "lora_name": lora_name,
        "storage_path": storage_path,
    }


runpod.serverless.start({"handler": handler})
