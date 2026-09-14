import hashlib
import json
from pathlib import Path

REGISTRY_PATH = Path("data/document_registry.json")

def calculate_file_hash(file_path: str) -> str:
    # SHA-256 lets us detect whether file content changed
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()

def load_registry() -> dict:
    # First ingestion has no registry yet
    if not REGISTRY_PATH.exists():
        return {}

    with open(REGISTRY_PATH, "r") as file:
        return json.load(file)

def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as file:
        json.dump(registry, file, indent=2)

def has_document_changed(file_path: str, registry: dict) -> bool:
    file_name = Path(file_path).name
    current_hash = calculate_file_hash(file_path)

    previous = registry.get(file_name)

    # New file
    if not previous:
        return True

    return previous.get("hash") != current_hash

def update_document_registry(file_path: str, registry: dict):
    file_name = Path(file_path).name

    registry[file_name] = {
        "hash": calculate_file_hash(file_path),
    }
def get_kb_version(registry: dict) -> str:
    # Combine all document hashes into one stable KB version
    combined_hashes = "".join(
        registry[file_name]["hash"]
        for file_name in sorted(registry)
    )

    return hashlib.sha256(
        combined_hashes.encode()
    ).hexdigest()