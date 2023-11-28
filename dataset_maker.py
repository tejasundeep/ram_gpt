import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

def generate_qa_pairs(text, tokenizer, model):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

    # Generate the question and answer
    outputs = model.generate(**inputs, max_length=100)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Use regular expression to separate question and answer
    match = re.match(r"(.*?)\? (.+)", question_answer)
    if match:
        question, answer = match.groups()
    else:
        print("Unexpected format:", question_answer)
        question, answer = "Unable to parse", "Unable to parse"

    return question, answer

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")

# Read text from a file
with open('test.txt', 'r', encoding='utf-8') as file:
    context = file.read()

# Split the context into smaller chunks
# Adjust the chunk size as needed
chunk_size = 100  # Example chunk size
chunks = [context[i:i + chunk_size] for i in range(0, len(context), chunk_size)]

qa_pairs = []
for chunk in chunks:
    question, answer = generate_qa_pairs(chunk, tokenizer, model)
    qa_pairs.append({"question": question, "answer": answer})

# Write JSON data to a file
with open('qa_output.json', 'w', encoding='utf-8') as json_file:
    json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)

print("Question and Answer JSON file created successfully.")
