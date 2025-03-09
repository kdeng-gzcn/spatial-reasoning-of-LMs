import os 
import json

import pandas as pd

JSON_PATH = 'Result/Pair VLM Exp on phi newbenchmark/1740621448/conversations.json'
CSV_PATH = 'Result/Pair VLM Exp on phi newbenchmark/1740621448/full_history.csv'
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

metadata = data[0]
data = data[1:]

final_csv = {
    "scene": [],
    "seq": [],
    "pair": [],
    "dof": [],
    "label": [],
    "degree": [],
    "rnd": [],
    "speaker": [],
    "listener": [],
    "text": [],
}

for each_conversation in data:
    conversation_metadata = each_conversation[0]
    scene = conversation_metadata.get('scene', None)
    seq = conversation_metadata.get('seq', None)
    pair = conversation_metadata.get('pair', None)
    dof = conversation_metadata.get('significant dof', None)
    label = conversation_metadata.get('label', None)
    degree = conversation_metadata.get('significant value', None)
    
    for each_sentence in each_conversation[1:]:
        rnd = each_sentence.get('round_num', None)
        speaker = each_sentence.get('speaker', None)
        listener = each_sentence.get('listener', None)
        text = each_sentence.get('text', None)

        final_csv["scene"].append(scene)
        final_csv["seq"].append(seq)
        final_csv["pair"].append(pair)
        final_csv["dof"].append(dof)
        final_csv["label"].append(label)
        final_csv["degree"].append(degree)
        final_csv["rnd"].append(rnd)
        final_csv["speaker"].append(speaker)
        final_csv["listener"].append(listener)
        final_csv["text"].append(text)

pd.DataFrame(final_csv).to_csv(CSV_PATH, index=False)
print(f"CSV file saved at {CSV_PATH}")
        