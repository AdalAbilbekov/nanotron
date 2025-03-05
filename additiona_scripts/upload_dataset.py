from huggingface_hub import HfApi, login

login(token="", add_to_git_credential=False)

api = HfApi()

api.create_repo(repo_id="issai/base_train_cleanes_v1", repo_type="dataset", exist_ok=True, private=True)

# api.upload_file(
#     path_in_repo="model_config.json",
#     path_or_fileobj="/scratch/adal_abilbekov/models/Llama_1.5B_10-02-2025-V1/209000/model_config.json",
#     repo_id="issai/instruct_bundle_cleaned",
# )

api.upload_large_folder(
    folder_path="/scratch/adal_abilbekov/ds_llm",
    repo_id="issai/base_train_cleanes_v1",
    repo_type="dataset"
)