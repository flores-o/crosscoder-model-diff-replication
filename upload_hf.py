from huggingface_hub import HfApi

# Define repository details
repo_id = "oanaflores/crosscoder-gemma-2-2b-model-diff-matryoshka-v2"
folder_path = "blocks.14.hook_resid_pre"

# Local file paths to upload
files_to_upload = {
    "/workspace/crosscoder-model-diff-replication/checkpoints/version_3/5_cfg.json": f"{folder_path}/5_cfg.json",
    "/workspace/crosscoder-model-diff-replication/checkpoints/version_3/5.pt": f"{folder_path}/5.pt",
}

# Initialize the Hugging Face API
api = HfApi()

# Upload files
for local_path, repo_path in files_to_upload.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="model"  # Change this to "dataset" if it's a dataset repository
    )
    print(f"Uploaded {local_path} to {repo_id}/{repo_path}")
