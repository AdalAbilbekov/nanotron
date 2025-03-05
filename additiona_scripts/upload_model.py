from huggingface_hub import HfApi, login

login(token="", add_to_git_credential=False)

api = HfApi()

api.create_repo(repo_id="issai/franky_pt", repo_type="model", exist_ok=True, private=True)

# api.upload_file(
#     path_in_repo="model_config.json",
#     path_or_fileobj="/scratch/adal_abilbekov/models/Llama_1.5B_10-02-2025-V1/421000/model_config.json",
#     repo_id="AdalAbilbekov/small_lmV1_421000",
# )

api.upload_folder(
    folder_path="/scratch/adal_abilbekov/models/LLama-3.1-KazLLM-1.0-8B-Sandwich/output",
    repo_id="issai/franky_pt",
    repo_type="model"
)