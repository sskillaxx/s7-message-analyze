from huggingface_hub import HfApi

api = HfApi()

REPO_ID = "dalture/s7-eng-topic"

api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path="/home/dalture/s7-project-loc/src/s7_project/models/finetuning/english_topic_model",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="upload finetuned eng-topic model"
)

# model link: https://huggingface.co/dalture/s7-eng-topic