import runpod
import subprocess, os, requests, zipfile, shutil

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
                sh
