import json
from haystack import Document
from haystack.components.readers import ExtractiveReader


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_questions_and_contexts(data):
    questions = []
    docs = []
    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            docs.append(Document(content=context))
            for qa in paragraph["qas"]:
                questions.append((qa["id"], qa["question"]))
    return questions, docs


def generate_predictions(questions, docs):
    reader = ExtractiveReader(model="deepset/roberta-base-squad2")
    reader.warm_up()
    predictions = {}

    for q_id, question in questions:
        result = reader.run(query=question, documents=docs)
        if result["answers"]:
            predictions[q_id] = result["answers"][0].data

        else:
            predictions[q_id] = ""

    return predictions


def save_predictions(predictions, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)


def main():
    input_file = "covid-qa-test.json"
    output_file = "predictions_test.json"

    data = load_data(input_file)
    questions, docs = extract_questions_and_contexts(data)

    predictions = generate_predictions(questions, docs)
    save_predictions(predictions, output_file)

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
