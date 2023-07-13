# from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# def generate_similar_sentence(prompt, model_name="EleutherAI/gpt-neo-1.3B", temperature=0.5):
#     model = GPTNeoForCausalLM.from_pretrained(model_name)
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     output = model.generate(input_ids, pad_token_id=model.config.eos_token_id, do_sample=True, min_length=len(prompt), max_length=len(prompt) + 5, num_return_sequences=1, temperature=temperature)

#     generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

#     return generated_sentence

# input_sentence = "lol, I don't think you'd be fucking over DeafPirateRoberts by doing that.  He was giving you a valid warning.  Use of a drop is a lot more difficult to pull off.  Past success is not a good predictor of future success in this game as all it takes is one returned to sender, one suspicious postman, etc etc.  Assume your drop is a vacant home.  If so, risky.\n\nGet a POBox, they are cheap."

# similar_sentence = generate_similar_sentence(input_sentence)

# print(similar_sentence)

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import os
import tqdm
# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the model to evaluation mode
model.eval()

# Encode input text


texts_train = []
labels_train = []

print("Loading data")
with open('darkreddit_authorship_attribution_train_anon.jsonl', 'r') as f:
    for line in f:
        line = json.loads(line)
        for key, value in line.items():
            if key == 'author':
                labels_train.append(value)
            elif key == 'comment':
                texts_train.append(value)
    f.close()
print("Loading finished")

print("Start augmentation")
path = "Data/train_augmentation10.jsonl"
# b = True
a = 0
if not os.path.exists(path):
    with open(path, 'w') as output_file:
        for i in tqdm.tqdm(range(len(labels_train)), desc="Train data augmentation10"):
            if labels_train[i] in ["user10"]:
                if a ==120:
                    break
                a += 1
                # if b:
                text = texts_train[i]
                input_text = texts_train[i]
                input_ids = tokenizer.encode(input_text, return_tensors="pt")
                attention_mask = input_ids != tokenizer.pad_token_id
                input_length = len(input_text.split())
                # Generate text
                if input_length*5>700:
                    continue
                print(input_length*5)
                output = model.generate(input_ids, max_length=input_length*5, num_return_sequences=1, temperature=1)
                # Decode generated text and extract a sentence with similar length to the input
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                sentences = generated_text.split(".")
                output_sentence = min(sentences, key=lambda x: abs(len(x.split()) - input_length))
                data = {
                    "author": labels_train[i],
                    "comment": output_sentence.strip()
                }
                json_str = json.dumps(data)
                output_file.write(json_str + "\n")
                # else:
                #     b = False
    output_file.close()
else:
    print("File exist")
print("Finish augmentation")





