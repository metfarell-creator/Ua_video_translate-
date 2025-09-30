import os
from pathlib import Path
from huggingface_hub import snapshot_download, login

MODELS = [
    "openai/whisper-large-v3",
    "patriotyk/styletts2-ukrainian",
]

def main():
    hf_token = os.getenv("HF_TOKEN") or ""
    hf_home = os.getenv("HF_HOME") or ".cache/huggingface"
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(Path(hf_home).absolute())

    if hf_token:
        try:
            login(token=hf_token)
            print("[INFO] HF login OK")
        except Exception as e:
            print("[WARN] HF login failed:", e)

    for repo in MODELS:
        try:
            print("[INFO] Downloading:", repo)
            snapshot_download(repo_id=repo, local_dir=None, local_dir_use_symlinks=False, tqdm_class=None)
        except Exception as e:
            print("[WARN] Prefetch failed for", repo, "->", e)

if __name__ == "__main__":
    main()
