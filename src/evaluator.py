

import argparse, sys, os, copy, time, random, json, pickle, re, collections
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from rouge_score import rouge_scorer
ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])




def get_instruction_suffix(args):
    if args.data in ['arithmetics']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 12.34}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 12.34}".'
    elif args.data in ['gsm8k']:
        if args.bae :
            return ' Make sure to state your answer at the end of the response.'
        elif args.cot :
            return " Make sure to state your final answer in curly brackets at the very end of your response, just like: '{final answer: 123}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
    elif args.data in ['hellaswag','pro_medicine','formal_logic','csqa','hh_rlhf']:
        if args.bae :
            return ' Put your final answer in the form (X) at the end of your response.'
        elif args.cot :
            return " Make sure to state your final answer choice in curly brackets at the very end of your response, just like: '{final answer: (A)}'. Let's think step by step."
        else :
            return ' Make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
    elif args.data in ['cnn_daily']:
        return ' Make sure to provide your summary after stating "# Summary # ".'



def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    # for _, response in responses.items():
    #     try:
    #         # 1. 找第一种格式
    #         pred_list = re.findall(r"\{final answer:\s*(.*?)\}", response)

    #         # 2. 如果没找到，找第二种格式
    #         if not pred_list:
    #             pred_list = re.findall(r"\\boxed\{(.*?)\}", response)
            
    #         # 3. 如果两种都没找到，主动触发异常跳入 except
    #         if not pred_list:
    #             raise ValueError("No answer found")
                
    #         # 4. 安全地取最后一位，清理多余空格，并转为数字
    #         pred_str = pred_list[-1]
    #         pred_float = float(pred_str.strip())
            
    #         # 5. 四舍五入并存入结果
    #         final_answers.append(np.round(pred_float, 1))
        
    #     except Exception as e:
    #     # 无论是没找到，还是 float() 转换失败（例如提取出了字母），都存入空值
    #     # print(f"解析失败: {e}") # 调试时可以打开这行，看看模型到底输出了什么奇葩格式
    #         final_answers.append("")
    import re
import numpy as np

final_answers = []
for _, response in responses.items():
    try:
        # 1. 找第一种格式（加入了 \\? 来优雅地处理可能出现的反斜杠）
        pred_list = re.findall(r"\\?\{final answer:\s*(.*?)\\?\}", response)

        # 2. 如果没找到，找第二种格式
        if not pred_list:
            pred_list = re.findall(r"\\+boxed\s*\{\s*(.*?)\s*\}", response)
        
        # 3. 如果两种都没找到，主动触发异常跳入 except
        if not pred_list:
            raise ValueError("No answer found")
            
        # 4. 安全地取最后一位
        pred_str = pred_list[-1]
        
        # --- 新增：稳健的文本清洗步骤 ---
        # 移除可能残留的反斜杠、大括号，或者清理掉像 Agent 4 那样嵌套生成的额外文本
        pred_str_clean = pred_str.replace('\\', '').replace('{', '').replace('}', '').strip()
        if "final answer:" in pred_str_clean:
            pred_str_clean = pred_str_clean.replace("final answer:", "").strip()
        
        # 5. 转换为浮点数
        pred_float = float(pred_str_clean)
        
        # 6. 四舍五入并存入结果
        final_answers.append(np.round(pred_float, 1))
    
    except Exception as e:
        # print(f"解析失败: {e}") # 调试时可以打开，查看具体的报错内容
        final_answers.append("")


    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)


def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = pred.replace("final answer:", "").strip()
            if len(pred) == 0 :
                final_answers.append("")
            elif len(pred) < 3 :
                pred = pred[0]
                final_answers.append(f"({pred})")
            else :
                pred = pred[1]
                final_answers.append(f"({pred})")
        except :
            final_answers.append("")
    
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    return final_answers, debate_answer, debate_answer == answer

def evaluate_gen(responses, answer):
    final_answers = []
    debate_answer = None
    best_score = 0
    for _, input_str in responses.items():
        summary = input_str.split("# Summary #")[-1]
        final_answers.append(summary)

        scores = ROUGE.score(answer, summary)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure

        if rougeL > best_score :
            debate_answer = summary 
            best_scores = (rouge1, rouge2, rougeL)
        
    return final_answers, debate_answer, best_scores

def base_evaluate_arithmetics(responses, answer):
    final_answers = []
    for _, sentence in responses.items():
        parts = sentence.split(" ")

        for part in parts[::-1]:
            try:
                ans = float(part)
                final_answers.append(ans)
                break
            except:
                continue

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = "" 

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)

def base_evaluate_mcq(responses, answer):

    final_answers = []
    for _, input_str in responses.items():

        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                final_answers.append(f"({solution})")
                break

    counter = collections.Counter([x for x in final_answers if x != ""])
    try :
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = ""
    return final_answers, debate_answer, debate_answer == answer




