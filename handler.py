import runpod
import subprocess, os, requests, zipfile, shutil

SUPABASE_URL = "https://dksemexxbmgmtdfbidnk.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def upload_to_supabase(local_path, storage_path):
    with open(local_path, "rb") as f:
        data = f.read()
    url = f"{SUPABASE_URL}/storage/v1/object/lora-models/{storage_path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/octet-stream",
    }
    r = requests.post(url, headers=headers, data=data)
    r.raise_for_status()
    return f"{SUPABASE_URL}/storage/v1/object/public/lora-models/{storage_path}"

def handler(event):
    inp = event["input"]
    dataset_url = inp["dataset_zip_url"]
    trigger      = inp["trigger_word"]
    steps        = inp.get("steps", 2000)
    lr           = inp.get("learning_rate", 1e-4)
    batch        = inp.get("batch_size", 1)
    rank         = inp.get("rank", 32)
    alpha        = inp.get("alpha", 16)
    lora_name    = inp.get("lora_name", f"{trigger}_lora")
    project_id   = inp.get("project_id", "unknown")

    zip_path = "/tmp/dataset.zip"
    r = requests.get(dataset_url); r.raise_for_status()
    with open(zip_path, "wb") as f: f.write(r.content)
    extract_dir = "/tmp/training/dataset"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z: z.extractall(extract_dir)

    kohya_dir = f"/tmp/training/dataset/1_{trigger}"
    os.makedirs(kohya_dir, exist_ok=True)
    for root, _, files in os.walk(extract_dir):
        if root == kohya_dir: continue
        for fname in files:
            if fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                shutil.move(os.path.join(root, fname), os.path.join(kohya_dir, fname))

    output_dir = "/tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "accelerate", "launch", "--num_cpu_threads_per_process=1",
        "/kohya/sdxl_train_network.py",
        f"--pretrained_model_name_or_path=/models/sdxl/sd_xl_base_1.0.safetensors",
        f"--train_data_dir=/tmp/training/dataset",
        f"--output_dir={output_dir}",
        f"--output_name={lora_name}",
        "--network_module=networks.lora",
        f"--network_dim={rank}", f"--network_alpha={alpha}",
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
        "--xformers",
        "--save_model_as=safetensors",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr[-2000:]}

    safetensors = os.path.join(output_dir, f"{lora_name}.safetensors")
    if not os.path.exists(safetensors):
        return {"error": "Output file not found"}

    storage_path = f"{project_id}/{lora_name}.safetensors"
    public_url = upload_to_supabase(safetensors, storage_path)

    return {
        "status": "completed",
        "lora_name": lora_name,
        "safetensors_url": public_url,
        "storage_path": storage_path
    }

runpod.start({"handler": handler})
