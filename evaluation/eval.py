import json
import os
import random
import re
import time
import warnings
import zipfile
from datasets import load_from_disk
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM ,AutoModelForCausalLM
from transformers import T5Tokenizer, MT5ForConditionalGeneration,T5ForSequenceClassification
import jieba
from nltk.translate.bleu_score import sentence_bleu
from peft import PeftModel,PeftConfig
# from rouge import Rouge
from rouge_chinese import Rouge



def int_llm():
    # 基座
    # model = AutoModelForCausalLM.from_pretrained('my_model/model_1',device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
    # model =model.eval()
    # tokenizer = AutoTokenizer.from_pretrained('my_model/model_1',use_fast=False)
    # finetune_model_path='my_model/lora_1'  
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    # config = PeftConfig.from_pretrained(finetune_model_path)
    # config.base_model_name_or_path = "my_model/model_1"
    model = AutoModelForCausalLM.from_pretrained('二阶段全微调_全诗词数据_1epoch',device_map=device_map,torch_dtype=torch.float16,load_in_8bit=False,trust_remote_code=True,use_flash_attention_2=False)
    model =model.eval()
    tokenizer = AutoTokenizer.from_pretrained('二阶段全微调_全诗词数据_1epoch',use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.eval()
    return model,tokenizer


def CUGE_GLM(model , enc):
    run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    # model , enc = int_local_model(out_dir)
    
    score = []
    with open("data/ccpm/val_data.json","r") as f:
        val_data = json.load(f)
    error_answer = []
    for example in tqdm(val_data):
        instruction = example["instruction"]
        prompt =example["prompt"]
        answer=example["output"]

        prompt = instruction + "\n" +prompt + "\n 回答："

        # prompt_tokens = enc.tokenizer.encode(prompt, bos=True, eos=False)
        inputs = enc(prompt, return_tensors="pt",add_special_tokens=False).input_ids
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        len_prompt = len(inputs)
        generate_input = {
            "input_ids":inputs,
            "max_new_tokens":len_prompt+1,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":enc.eos_token_id,
            "bos_token_id":enc.bos_token_id,
            "pad_token_id":enc.pad_token_id
        }
        outputs = model.generate(**generate_input)
        out = enc.decode(outputs[0].tolist())
#        print(out)
#        pred = out[len(prompt)+1]
        pred = out[len(prompt)]
#        print(pred)
        correct = 1 if pred == answer else 0
        score.append(correct)
    correct_ratio = 100*sum(score)/len(score)
    print("acc:",correct_ratio)

def geshi(model , enc):

    score = []
    for i in tqdm(range(100)):
        gehshis = ["五言绝句","七言绝句","五言律诗","七言律诗"]
        geshi_choice = random.choice(gehshis)
        prompt = "写一首诗，要求："+ geshi_choice + "。回答："
        inputs = enc(prompt, return_tensors="pt",add_special_tokens=False).input_ids
        len_prompt = len(inputs)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        generate_input = {
             "input_ids":inputs,
             "max_new_tokens":512,
             "do_sample":True,
             "top_k":5,
             "top_p":0.95,
             "temperature":0.3,
             "repetition_penalty":1.3,
             "eos_token_id":enc.eos_token_id,
             "bos_token_id":enc.bos_token_id,
             "pad_token_id":enc.pad_token_id
        }
        outputs = model.generate(**generate_input)
        # print(outputs)
        pt_tokens = outputs[0].tolist()
        pt_tokens = pt_tokens[len_prompt:]
        to_remove = {55296, 55297, 55298, 55299}
#
#        # 使用列表推导式移除指定元素
        pt_tokens = [x for x in pt_tokens if x not in to_remove]
        try:
            print(enc.decode(pt_tokens))
        
            index = pt_tokens.index(enc.eos_token_id)
        except:
            continue
        chunk1 = pt_tokens[:index]
        # 提取模型回答
        output = enc.decode(chunk1)
        output_=output[len(prompt):]
        output_ = output_.replace(" ", "")
#        print(geshi_choice,output_)
        
        ticai = ""
        sentences  = [sentence for sentence in re.split("[，。？！]", output_) if sentence != ""]  
        if len(sentences) == 4:
            if all([len(sentence) == 5  for sentence in sentences]):
                ticai = "五言绝句"
            elif all([len(sentence) == 7  for sentence in sentences]):
                ticai = "七言绝句"
        elif len(sentences) == 8:
            if all([len(sentence) == 5  for sentence in sentences]):
                ticai = "五言律诗"
            elif all([len(sentence) == 7  for sentence in sentences]):
                ticai = "七言律诗"
#        print(ticai)
        correct = 1 if geshi_choice == ticai else 0
        score.append(correct)
    correct_ratio = 100*sum(score)/len(score)
    print("格式准确度：",correct_ratio)


def ROUGE(model , enc):
    
    score_all = []
    with open("data/vaild_ccpc_以关键字生成诗词.json","r") as f:
        val_data = json.load(f)
    val_data = val_data[:300]
    error_answer = []
    for example in tqdm(val_data):
        instruction = example["instruction"]
        prompt =example["prompt"]
        answer=example["output"]
        prompt = instruction + "\n" +prompt + "\n回答:"
        inputs = enc(prompt, return_tensors="pt",add_special_tokens=False).input_ids
        len_prompt = len(inputs)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        generate_input = {
             "input_ids":inputs,
             "max_new_tokens":512,
             "do_sample":True,
             "top_k":5,
             "top_p":0.95,
             "temperature":0.3,
             "repetition_penalty":1.3,
             "eos_token_id":enc.eos_token_id,
             "bos_token_id":enc.bos_token_id,
             "pad_token_id":enc.pad_token_id
        }
        outputs = model.generate(**generate_input)
        # print(outputs)
        pt_tokens = outputs[0].tolist()
        pt_tokens = pt_tokens[len_prompt:]
#        print(pt_tokens,enc.decode(pt_tokens))

        try:
            index = pt_tokens.index(enc.eos_token_id)
        except:
            continue
        chunk1 = pt_tokens[:index]
        # 提取模型回答
        output = enc.decode(chunk1)
        output_=output[len(prompt):]    
        rouge = Rouge()
        reference = ' '.join(jieba.cut(answer, cut_all=True))
        references = [reference]
        candidate = ' '.join(jieba.cut(output_, cut_all=True))
        # print(reference,"|||",candidate)
        try:
            score = rouge.get_scores(candidate, reference)
        except:
            continue
        
        score1 = score[0]["rouge-1"]["f"]
        # print(score1)
        score_all.append(score1)
    correct_ratio = sum(score_all)/len(score_all)
    print("ROUGE:",correct_ratio)


def perplexity(model , enc):
    import math
    score_all = []
    with open("data/vaild_ccpc_以关键字生成诗词.json","r") as f:
        val_data = json.load(f)
    val_data = val_data[:300]
    error_answer = []
    total_loss = 0
    total_words = 0
    for text in tqdm(val_data):
        instruction = text["instruction"]
        prompt =text["prompt"]
        answer=text["output"]
        prompt = instruction + "\n" +prompt + "\n回答:" + answer
        inputs = enc(prompt, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        num_words = inputs["input_ids"].size(1)  # 当前句子的词数
        total_loss += loss.item() * num_words
        total_words += num_words

    # 数据集的平均困惑度
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    print(f"Dataset Perplexity: {perplexity}")


def D_N(model , enc):
    score = []

    with open("data/vaild_ccpc_以关键字生成诗词.json","r") as f:
        val_data = json.load(f)
    val_data = val_data[:300]
    error_answer = []
    for example in tqdm(val_data):
        instruction = example["instruction"]
        prompt =example["prompt"]
        answer=example["output"]
        prompt = instruction + "\n" +prompt
        inputs = enc(prompt, return_tensors="pt",add_special_tokens=False).input_ids
        len_prompt = len(inputs)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        generate_input = {
             "input_ids":inputs,
             "max_new_tokens":512,
             "do_sample":True,
             "top_k":5,
             "top_p":0.95,
             "temperature":0.3,
             "repetition_penalty":1.3,
             "eos_token_id":enc.eos_token_id,
             "bos_token_id":enc.bos_token_id,
             "pad_token_id":enc.pad_token_id
        }
        outputs = model.generate(**generate_input)
        # print(outputs)
        pt_tokens = outputs[0].tolist()
        pt_tokens = pt_tokens[len_prompt:]
        # print(pt_tokens,enc.decode(pt_tokens))
        try:
            index = pt_tokens.index(enc.eos_token_id)
        except:
            continue
        chunk1 = pt_tokens[:index]
        # 提取模型回答
        output = enc.decode(chunk1)
        output_=output[len(prompt):]
        reference = ' '.join(jieba.cut(answer))
        candidate = ' '.join(jieba.cut(output_))
        references = [reference,candidate]
        try:
            score1 = calc_distinct_k(references,2)
        except:
            continue
        # print(score1)
        score.append(score1)
    correct_ratio = sum(score)/len(score)
    print("acc:",correct_ratio)


def GPT_eval(model , enc):


    from openai import OpenAI
    score_totals = {"流畅程度": 0, "含义": 0, "一致性": 0, "相关性": 0, "美学": 0}
    count = 0
    for i in tqdm(range(100)):
        gehshis = ["五言绝句","七言绝句","五言律诗","七言律诗"]
        ticais = [
                "风流", "功名", "富贵", "何事", "无事", "往事", "心事", "归去", "归来", 
                "回首", "别离", "相逢", "相见", "不见", "一笑", "不知", "谁知", "不得", 
                "不似", "殷勤", "相思", "多情", "无情", "惆怅", "寂寞", "断肠", "肠断", 
                "可怜", "明月", "风月", "日月", "东君", "白日", "斜阳", "夕阳", "落日", 
                "秋风", "清风", "青山", "青云", "白云", "浮云", "烟霞", "流水", "春色"
        ]

        geshi_choice = random.choice(gehshis)
        ticai_choice = random.choice(ticais)
        prompt = "写一首关于"+ticai_choice+"的诗，要求："+ geshi_choice + "。回答："
        inputs = enc(prompt, return_tensors="pt",add_special_tokens=False).input_ids
        len_prompt = len(inputs)
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        generate_input = {
             "input_ids":inputs,
             "max_new_tokens":512,
             "do_sample":True,
             "top_k":5,
             "top_p":0.95,
             "temperature":0.3,
             "repetition_penalty":1.3,
             "eos_token_id":enc.eos_token_id,
             "bos_token_id":enc.bos_token_id,
             "pad_token_id":enc.pad_token_id
        }
        outputs = model.generate(**generate_input)
        # print(outputs)
        pt_tokens = outputs[0].tolist()
        output = enc.decode(pt_tokens)
        
        message = "请分别从流畅程度，含义，一致性，相关性和美学五个角度对下面针对指令生成的诗歌进行评分（0-5），回答不需要明确的回答，请给出字典格式的打分：\n"+output
    
        client = OpenAI(
    	  api_key= "sk-PaQL17OGKeBMgYA4Eb5d567fCf9146BaB0Ad7eD0F466Be25",
    	  base_url  = "https://api.v3.cm/v1/",
    	  default_headers = {"x-foo": "true"}
    	)

        completion = client.chat.completions.create(
    	  model="gpt-4o-mini",
    	  messages=[
        {"role": "user", "content": message}
    	  ]
    	)
    	   # 初始化一个字典，用于累积每个评分项的总分

        
        try:
            response = completion.choices[0].message
            response_content = response.content
            
    	       # 将 JSON 字符串解析为 Python 字典
            scores = json.loads(response_content)
            for key, value in scores.items():
            	score_totals[key] += value
            # 提取评分值，存入数组
            
            count += 1
            print(score_totals,count )
        except json.JSONDecodeError as e:
            print(f"解析 JSON 数据时出错: {e}")

        
        

        # 计算平均值
    average_scores = {key: total / count for key, total in score_totals.items()}
    print(count)
    print(f"累积评分: {score_totals}")
    print(f"平均评分: {average_scores}")

      




def calc_distinct_k(a, k):
    d = {}
    tot = 0
    for sen in a:
        for i in range(0, len(sen)-k):
            key = tuple(sen[i:i+k])
            d[key] = 1
            tot += 1
    if tot > 0:
        dist = len(d) / tot
    else:
        warnings.warn('the distinct is invalid')
        dist = 0.
    return dist

model , enc = int_llm()
#
#GPT_eval(model , enc)


#perplexity(model , enc)
#CUGE_GLM(model , enc)
geshi(model , enc)
#ROUGE(model , enc)
#D_N(model , enc)


# from rouge_chinese import Rouge
# import jieba # you can use any other word cutting library

# hypothesis = "###刚刚发声，A股这种情况十分罕见！大聪明逆市抄底330亿，一篇研报引爆全球，市场逻辑生变？"
# hypothesis = ' '.join(jieba.cut(hypothesis)) 

# reference = "刚刚过去的这个月，美股总市值暴跌了将近6万亿美元（折合人民币超过40万亿），这背后的原因可能不仅仅是加息这么简单。最近瑞士信贷知名分析师Zoltan Polzsar撰写了一篇极其重要的文章，详细分析了现有世界秩序的崩坏本质以及美国和西方将要采取的应对策略。在该文中，Zoltan Polzsar直指美国通胀的本质和其长期性。同期，A股市场亦出现了大幅杀跌的情况。"
# reference = ' '.join(jieba.cut(reference))

# rouge = Rouge()
# scores = rouge.get_scores(hypothesis, reference)
# print(scores)
