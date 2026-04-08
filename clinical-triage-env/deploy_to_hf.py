"""Deploy ClinicalTriageEnv to HuggingFace Spaces."""

from huggingface_hub import HfApi, create_repo
import os

REPO_ID = "Nermal007/clinical-triage-env"
SPACE_SDK = "docker"

api = HfApi()

# 1. Create the Space (Docker SDK)
print("Creating HuggingFace Space...")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk=SPACE_SDK,
        exist_ok=True,
    )
    print(f"Space created: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"Space may already exist: {e}")

# 2. Upload all project files (excluding .env, test script, and this deploy script)
print("\nUploading files...")

project_dir = os.path.dirname(os.path.abspath(__file__))

EXCLUDE = {".env", "deploy_to_hf.py", "test_smoke.py", ".git", "__pycache__", ".venv"}

files_uploaded = 0
for root, dirs, files in os.walk(project_dir):
    # Skip excluded directories
    dirs[:] = [d for d in dirs if d not in EXCLUDE]
    
    for fname in files:
        if fname in EXCLUDE:
            continue
        if fname.endswith(".pyc"):
            continue
            
        local_path = os.path.join(root, fname)
        # Path relative to project dir
        rel_path = os.path.relpath(local_path, project_dir).replace("\\", "/")
        
        print(f"  Uploading: {rel_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=rel_path,
            repo_id=REPO_ID,
            repo_type="space",
        )
        files_uploaded += 1

print(f"\n✓ Uploaded {files_uploaded} files")
print(f"\n🚀 Your Space is live at: https://huggingface.co/spaces/{REPO_ID}")
print(f"   It will take 2-3 minutes to build the Docker image.")
print(f"\n   Once built, the API will be at:")
print(f"   https://{REPO_ID.replace('/', '-').lower()}.hf.space/health")
