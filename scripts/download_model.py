import os
import argparse
from huggingface_hub import snapshot_download


def verify_model_files(save_path: str) -> None:
    """
    Verify that all required model files exist in the specified directory.

    :param save_path: Path to the directory where the model is saved.
    :raises FileNotFoundError: If any required file is missing.
    """
    required_files: list[str] = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
    existing_files: list[str] = os.listdir(save_path)
    missing_files: list[str] = [file for file in required_files if file not in existing_files]

    if missing_files:
        raise FileNotFoundError(
            f"The following required files are missing in '{save_path}': {', '.join(missing_files)}. "
            f"Ensure the model repository is complete."
        )
    print(f"All required files are present in '{save_path}':")
    for file in existing_files:
        print(file)


def download_model_offline(repo_id: str, save_path: str) -> None:
    """
    Download a Hugging Face model repository locally using snapshot_download.

    :param repo_id: The Hugging Face repository ID (e.g., "codellama/Llama-2-7b-hf").
    :param save_path: The local directory to save the model repository.
    """
    try:
        print(f"Downloading model repository: {repo_id}")
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

        # Download the model repository
        snapshot_download(repo_id=repo_id, repo_type="model", local_dir=save_path)

        # Verify that all required files are downloaded
        verify_model_files(save_path)

        print(f"Model repository saved successfully at: {save_path}")
    except Exception as e:
        print(f"Error downloading the model repository: {e}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for repo_id and save_path.

    :return: Parsed arguments namespace.
    """
    default_repo_id: str = "codellama/CodeLlama-7b-Instruct-hf"
    default_save_path: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../models/llama-2-7b")
    )
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repository for offline use."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=default_repo_id,
        help=f"Hugging Face repository ID (default: {default_repo_id})"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help=f"Local directory to save the model (default: {default_save_path})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_model_offline(args.repo_id, args.save_path)

    # Instructions for offline usage
    print("\nTo use the model offline, set the environment variable:")
    print("HF_HUB_OFFLINE=1")