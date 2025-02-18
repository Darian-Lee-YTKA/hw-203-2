import json
import torch
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from tqdm import tqdm  


data = {
    "test": "covid-qa-test.json",
    "dev": "covid-qa-dev.json"
}


gpu_id = 0 if torch.cuda.is_available() else -1
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


model_name = "./finetuned_roberta"  
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=gpu_id)


for split, file_path in data.items():
    print(f"\nProcessing {split} dataset...")


    with open(file_path, "r", encoding="utf-8") as f:
        covid_qa_data = json.load(f)


    qa_inputs = [
        {"question": qa["question"], "context": paragraph["context"], "id": qa["id"]}
        for entry in tqdm(covid_qa_data["data"], desc=f"Loading {split} dataset", unit="file")
        for paragraph in entry["paragraphs"]
        for qa in paragraph["qas"]
    ]


    batch_size = 16  
    predictions = {}

    print(f"Running inference on {len(qa_inputs)} samples from {split} dataset...")

    for i in tqdm(range(0, len(qa_inputs), batch_size), desc=f"Inference ({split})", unit="batch"):
        batch = qa_inputs[i : i + batch_size]
        results = qa_pipeline(
            question=[q["question"] for q in batch],
            context=[q["context"] for q in batch],
            batch_size=batch_size
        )


        for qa, result in zip(batch, results):
            predictions[qa["id"]] = result["answer"]


    output_file = f"REALfinetuned_model_predictions_{split}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    print(f"WE DONE BITCHESSSS!!!! Predictions saved to {output_file}")