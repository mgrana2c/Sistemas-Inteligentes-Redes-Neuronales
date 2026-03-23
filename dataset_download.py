from pathlib import Path

import kagglehub

project_dir = Path(__file__).resolve().parent
dataset_dir = project_dir / "dataset"
dataset_dir.mkdir(exist_ok=True)

path = kagglehub.dataset_download(
	"olistbr/brazilian-ecommerce",
	output_dir=str(dataset_dir),
)

print("Path to dataset files:", path)