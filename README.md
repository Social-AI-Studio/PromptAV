# Who Wrote it and Why? Prompting Large-Language Models for Authorship Verification

This is the official repository for our paper: Chia-Yu Hung, Zhiqiang Hu, Yujia Hu, Roy Ka-Wei Lee. [Who Wrote it and Why? Prompting Large-Language Models for Authorship Verification](https://arxiv.org/abs/2310.08123). EMNLP 2023 Findings.


In this work, we proposed PromptAV, a novel technique that leverage LLMs for authorship verification task that provide step-by-step stylometric explaination prompts. 


To run our experiment, simply clone this repository, install the requirements and execute the code block. The prompt folders contains the different prompts we used in our paper.
```
python ChatGPT_promptAV.py --output_dir="output_dir" \
    --dataset="AV_confidence" \
    --data_path="data/AV_test_1k.csv" \
    --api_key='YOUR-API-KEY' \
    --prompt='prompt/0shotPromptAV_prompt.json'
```
