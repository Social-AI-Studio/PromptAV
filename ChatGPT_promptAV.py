import openai
import json
import time
import re
import os 
import torch
from tqdm import tqdm
import argparse
import pandas as pd

def main():
    args = parse_args()
    dataset = args.dataset
    data_path = args.data_path
    prompt = args.prompt
    #data_path = f"dataset/{dataset}/test.json"

    api_list = [args.api_key]    
    output_folder = f"{dataset}/"
    output_file_name = args.output_dir
    output_file = f"{output_folder}{output_file_name}_chatgpt.json"
    with open(args.prompt, 'r') as f:
        for line in f:
            sample = json.loads(line)
    prompt = sample['prompt']
    prompt_end = sample['prompt_end']

    correct = 0

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if os.path.exists(output_file):
        json_data = []
        with open(output_file, 'r') as f:
            for line in f:
                sample = json.loads(line)
                json_data.append(sample)
                if sample["flag"]==True:
                    correct += 1
        start_point = len(json_data)
        print(f"The generated samples reloaded, the number of sample is {start_point}. The accuracy is {correct/start_point}.")
    else:
        json_data = []
        start_point = 0

    #text_list, label_list, ids = reader(data_path, dataset) 
    
    ## Read data from csv file instead of json file.
    text_list, label_list, ids = csv_reader(data_path, dataset)

    api_idx = 0
    openai.api_key = api_list[api_idx]

    for i in tqdm(range(start_point, len(text_list))):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        # messages.append({"role": "user", "content": 'Q: ' + text_list[i] + '\nA: The answer (arabic numerals) is'})
        if dataset == "winogrande":
            messages.append({"role": "user", "content": text_list[i] + "\n\nYour answer has to be one of the options: option 1, option 2"})
        else:
            messages.append({"role": "user", "content": prompt+'\n'+text_list[i]+'\n'+prompt_end})
        assistant, get_result = get_answer(messages)
        if not get_result:
            continue
        messages.append({"role": "assistant", "content": assistant})

        predict = extract_answer(args, assistant.lower())
        
        flag = False
        if predict == label_list[i]:
            correct += 1
            flag = True
        gen_data = {
            'ID': ids[i],
            'messages': messages,
            'prediction': predict,
            'label': label_list[i],
            'flag': flag,
            'raw': assistant.lower()
        }
        write_json(gen_data, output_file)
        print(f"ID: {ids[i]}, Prediction: {predict}, Label: {label_list[i]}, Flag: {flag}, Accuracy: {correct/(i+1)}")
    print("Finding max score...")
    df = pd.read_json(output_file,lines=True)
    max_score = find_max_score(df)
    print(f"Max score is {max_score}")
    

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["AV_confidence"],
                        required=True)
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--prompt',required=True)
    parser.add_argument('--output_dir',required=True)
    return parser.parse_args()

def reader(path, dataset):
    with open(path, 'r') as f:
        data=json.load(f)
    text_list = []
    label_list = []
    for sample in data:
        text_list.append(sample['instruction'])
        label_list.append(sample['answer'])
    ids = [i for i in range(len(text_list))]
    return text_list, label_list, ids


def basic_runner(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0
    )
    
    return response.get('choices', [{}])[0].get('message', {}).get('content', '')


def get_answer(messages):
    get_result = False
    retry = 0
    while not get_result:
        try:
            assistant = basic_runner(messages)
            get_result = True
            retry = 0
        except openai.error.RateLimitError as e:
            if e.user_message == 'You exceeded your current quota, please check your plan and billing details.':
                api_idx += 1 
                assert api_idx < len(api_list), "No available API keys!"
                apikey = api_list[api_idx]
                openai.api_key = apikey
                retry = 0
            elif retry < 3:
                time.sleep(2)
                retry += 1
            else:
                api_idx += 1 
                assert api_idx < len(api_list), "No available API keys!"
                apikey = api_list[api_idx]
                openai.api_key = apikey
                retry = 0
    return assistant, get_result

def extract_answer(args, sentence: str) -> str:
    dataset = args.dataset
    if dataset == 'AV_confidence':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'\b0?\.\d+\b', sentence_) ## This regex extract the confidence score
        if not pred_answers: ## Default answer no
            return "no"
        out = assign_label(pred_answers[-1].replace(' ','')) ## We use the last score as the confidence score
        return out
    
def extract_confidence_score(sentence):
    
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'\b0?\.\d+\b', sentence_) ## This regex extract the confidence score
    if not pred_answers: ## Default answer no
        return ""
    out = pred_answers[-1].replace(' ','') ## Assign yes or no base on the threshhold
    return out

def find_max_score(df):
    threshold_list = [i/10 for i in range(1,10)]
    score_list = []
    for threshold in threshold_list:
        extracted = df.raw.apply(extract_confidence_score)
        score_list.append(sum(extracted.apply(lambda x:assign_label(x,threshold)) == df.label)/len(df))
    
    return max(score_list)

def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, ensure_ascii=False)
    f.write('\n')
    f.close()

def csv_reader(path,dataset):
    
    df = pd.read_csv(path)
    text_list = df['combined_text'].to_list()
    label_list = df['labels'].to_list()
    ids = [i for i in range(len(text_list))]
    
    return text_list, label_list, ids

def assign_label(x,threshold=0.4):
    if x == '':
        return 'no'
    if float(x)<=threshold:
        return 'no'
    return 'yes'

if __name__ == "__main__":
    main()


