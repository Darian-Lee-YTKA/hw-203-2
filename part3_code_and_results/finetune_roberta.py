import json
from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import pynvml
import gc
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoConfig
from datasets import Dataset, DatasetDict
# Initialize NVIDIA management library
pynvml.nvmlInit()

device = torch.device("cuda:0" if torch.cuda.memory_allocated(0) < torch.cuda.memory_allocated(1) else "cuda:1")

from datasets import load_dataset


import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def convert_to_squadv2(train_data, test_data):
    squadv2_data = {"train": [], "test": []}
    
    # Обрабатываем тренировочные данные
    for article in train_data["data"]:
        title = article.get("title", "")
        
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            
            for qa in paragraph["qas"]:
                question = qa["question"]
                question_id = qa["id"]
                is_impossible = qa["is_impossible"]
                
                # Если is_impossible True, ответы пустые
                if is_impossible:
                    answers = []
                else:
                    answers = [
                        {"text": answer["text"], "answer_start": answer["answer_start"]}
                        for answer in qa["answers"]
                    ]
                
                squadv2_data["train"].append({
                    "id": question_id,
                    "title": title,
                    "context": context,
                    "question": question,
                    "answers": answers
                })

    # Обрабатываем тестовые данные
    for article in test_data["data"]:
        title = article.get("title", "")
        
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            
            for qa in paragraph["qas"]:
                question = qa["question"]
                question_id = qa["id"]
                is_impossible = qa["is_impossible"]
                
                # Если is_impossible True, ответы пустые
                if is_impossible:
                    answers = []
                else:
                    answers = [
                        {"text": answer["text"], "answer_start": answer["answer_start"]}
                        for answer in qa["answers"]
                    ]
                
                squadv2_data["test"].append({
                    "id": question_id,
                    "title": title,
                    "context": context,
                    "question": question,
                    "answers": answers
                })
    
    # Преобразуем данные в Hugging Face Dataset
    train_dataset = Dataset.from_dict({k: [d[k] for d in squadv2_data["train"]] for k in squadv2_data["train"][0]})
    test_dataset = Dataset.from_dict({k: [d[k] for d in squadv2_data["test"]] for k in squadv2_data["test"][0]})
    
    # Возвращаем в формате DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    return dataset_dict


    



train_file_path = "/home/daeilee/203-hw2/covid-qa-train.json"
val_file_path = "/home/daeilee/203-hw2/covid-qa-dev.json"
train_data = load_json(train_file_path)
val_data = load_json(val_file_path)
covid_squad = convert_to_squadv2(train_data, val_data)



print("Train Data:", covid_squad)  

from transformers import AutoTokenizer

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)




def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        if not answer:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        if "answer_start" not in answer[0]:
            print("🍏🍏ANSWER START NOT FOUND🍏🍏")
            print(answer)
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer[0]["answer_start"]
        end_char = answer[0]["answer_start"] + len(answer[0]["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context in the input sequence
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        context = examples["context"][i]  # Full context as string

        # Define the stride length and sliding window size
        window_size = 512
        stride = 256  # Half of the window size to ensure overlap

        # Loop over context to create multiple windows
        windows = []
        for start in range(context_start, context_end + 1, stride):
            end = min(start + window_size, context_end)
            if start >= context_end:
                break
            
            window = context[start:end]
            windows.append((start, end, window))

        # For each window, check if the answer fits
        valid_start, valid_end = None, None
        for start, end, window in windows:
            if start_char >= start and end_char <= end:
                # Find the token positions for this window
                input_start = start
                input_end = end

                start_positions.append(input_start)
                end_positions.append(input_end)
                valid_start, valid_end = start_positions, end_positions
                break

        if valid_start is None or valid_end is None:
            # If no valid window found, exclude this instance
            start_positions.append(0)
            end_positions.append(0)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
lora_config = LoraConfig(
    r=8,  
    lora_alpha=8,  
    lora_dropout=0.2,  
    bias="none",  
    task_type=TaskType.QUESTION_ANS 
)
model = get_peft_model(model, lora_config)

print(f"🍒🍒🍒The model is on: {model.device}")

tokenized_squad = covid_squad.map(preprocess_function, batched=True, remove_columns=covid_squad["train"].column_names)



training_args = TrainingArguments(
    output_dir="my_awesome_qa_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

MODEL_SAVE_PATH = "./finetuned_roberta"
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)