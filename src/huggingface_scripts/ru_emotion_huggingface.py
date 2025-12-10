from huggingface_hub import HfApi

api = HfApi()

REPO_ID = "dalture/s7-ru-emotions"

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path="src/models/ru_emotions",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="upload finetuned ru-emotions model"
)

# model link: https://huggingface.co/dalture/s7-ru-emotions