from langsmith import Client

from advanced_rag_agent.evaluation.dataset import EVALUATION_DATASET

DATASET_NAME = "agentic-rag-hr-evaluation"

client = Client()

# Reuse the dataset if it already exists
existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))

if existing_datasets:
    dataset = existing_datasets[0]
    print(f"Dataset already exists: {DATASET_NAME}")
else:
    # Create the LangSmith dataset only once
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for the Agentic RAG HR Assistant.",
    )
    print(f"Created dataset: {DATASET_NAME}")

# Convert our local test cases into LangSmith examples
inputs = []
outputs = []

for example in EVALUATION_DATASET:
    inputs.append({
        "question": example["question"],
    })

    outputs.append({
        "reference_answer": example["reference_answer"],
        "relevant_sections": example["relevant_sections"],
    })

client.create_examples(
    dataset_id=dataset.id,
    inputs=inputs,
    outputs=outputs,
)

print(f"Uploaded {len(EVALUATION_DATASET)} evaluation examples.")