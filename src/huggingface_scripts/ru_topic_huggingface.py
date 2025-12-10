from huggingface_hub import HfApi

api = HfApi()

REPO_ID = "dalture/s7-ru-topics"

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path="src/models/ru_topics",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="upload finetuned ru-topics model"
)

# model link: https://huggingface.co/dalture/s7-ru-topics