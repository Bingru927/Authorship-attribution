import json
import os
import tqdm
from googletrans import Translator

texts_train = []
labels_train = []

print("Loading data")
with open('darknet_authorship_verification_train_nodupe_anon.jsonl', 'r') as f:
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
path = "train_augmentation.jsonl"
# b = True
a = 0
if not os.path.exists(path):
    with open(path, 'w') as output_file:
        for i in tqdm.tqdm(range(len(labels_train)), desc="Train data augmentation6"):
            if labels_train[i] in ["user6"]:
                if a == 100:
                    break
                a += 1
                # if b:
                text = texts_train[i]
                translator = Translator()
                translations_first = translator.translate(text, dest='ko', src='auto')
                translations_second = translator.translate(translations_first.text, dest='fr', src='auto')
                translations_third = translator.translate(translations_second.text, dest='en', src='auto')
                data = {
                    "author": labels_train[i],
                    "comment": translations_third.text
                }
                json_str = json.dumps(data)
                output_file.write(json_str + "\n")
                # else:
                #     b = False
    output_file.close()
else:
    print("File exist")
print("Finish augmentation")
