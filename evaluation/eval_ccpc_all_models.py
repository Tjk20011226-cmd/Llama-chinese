# eval_all_models.py
# 批量对 Table1 中所有本地模型，在 CCPC 上计算：
#   - ROUGE-1 (mean ± std over seeds)
#   - BERTScore F1 (mean ± std over seeds)
#   - Distinct-1 / Distinct-2（内部多样性，按字）
#
# 生成结果保存到 results_rouge_bertscore.json，方便你更新 Table 1

import os
import json
import random
import numpy as np
from typing import Dict, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_chinese import Rouge
from bert_score import score as bert_score
import jieba
from tqdm import tqdm
import time

FAST_EVAL = True      # True → fast模式只跑50条
FAST_SAMPLES = 50
NORMAL_SAMPLES = 300


# ========= 配置区域 =========

CCPC_JSON = "/home/chengcheng/data/Llama-Chinese/data/vaild_ccpc_以关键字生成诗词.json"

# Table 1 中各模型与本地路径的映射
MODEL_PATHS = {
    "Chatglm3-6B": "/home/chengcheng/data/Llama-Chinese/chatglm3/chatglm3",
    "Qwen2-7B": "/home/chengcheng/data/Llama-Chinese/qwen2_7B",
    # 你机器上是 phi-2，如果以后换成 phi-3.5-mini，这里改路径即可
    "Phi-2": "/home/chengcheng/data/Llama-Chinese/phi-2",
    "Gemma-2-2B-it": "/home/chengcheng/data/Llama-Chinese/gemma-2-2b-it",
    "GLM-Edge-1.5B": "/home/chengcheng/data/Llama-Chinese/glm-1.5B",
    "Qwen2.5-0.5B-Instruct": "/home/chengcheng/data/Llama-Chinese/qwen2.5-0.5B-instruct",
    "Qwen2-0.5B-Instruct": "/home/chengcheng/data/Llama-Chinese/Qwen2-0.5B-Instruct",
    "MiniPoet-0.3B": "/home/chengcheng/data/Llama-Chinese/二阶段全微调_全诗词数据_1epoch",
}

# 统一生成参数（和你原来 eval.py 基本一致）
GEN_KWARGS = dict(
    max_new_tokens=512,
    do_sample=True,
    top_k=5,
    top_p=0.95,
    temperature=0.3,
    repetition_penalty=1.3,
)


# ========= 工具函数 =========

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path: str):
    """
    统一加载模型：
      - 大部分模型: torch_dtype="auto"
      - ChatGLM / GLM: torch.float16 （官方默认）
    """
    print(f"\n[Load] Loading model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        use_fast=False,
    )

    # 根据路径名简单判断一下是否是 GLM 系列
    lower_path = path.lower()
    if "chatglm" in lower_path or "glm-" in lower_path:
        dtype = torch.float16
    else:
        dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # pad_token 统一设置
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer



def eval_ccpc_one_seed(model, tokenizer, seed: int, max_samples: int = 300) -> Dict[str, float]:
    set_random_seed(seed)

    with open(CCPC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:max_samples]

    references = []
    candidates = []

    t0 = time.time()

    for idx, example in enumerate(data):

        # --- ETA 显示 ---
        elapsed = time.time() - t0
        speed = (idx + 1) / elapsed
        remaining = (len(data) - (idx + 1)) / speed if speed > 0 else 9999
        print(f"\rseed={seed} {idx+1}/{len(data)} | ETA: {remaining/60:.1f} min", end="", flush=True)

        try:
            instruction = example["instruction"]
            prompt = example["prompt"]
            answer = example["output"].strip()

            full_prompt = instruction + "\n" + prompt + "\n回答:"

            inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).input_ids
            len_prompt = inputs.shape[1]
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            generate_kwargs = {
                **GEN_KWARGS,
                "input_ids": inputs,
                "eos_token_id": tokenizer.eos_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }

            with torch.no_grad():
                outputs = model.generate(**generate_kwargs)

            pt_tokens = outputs[0].tolist()[len_prompt:]

            # 截 eos
            try:
                eos = pt_tokens.index(tokenizer.eos_token_id)
                pt_tokens = pt_tokens[:eos]
            except ValueError:
                pass

            output_text = tokenizer.decode(pt_tokens).strip()

            references.append(answer)
            candidates.append(output_text)

        except Exception as e:
            print(f"\n[Error sample idx={idx}, seed={seed}] {e}")
            continue

    print()  # 换行

    # ==== 评价 ====
    if not candidates:
        return {"rouge1": 0, "bertscore_f1": 0, "distinct1": 0, "distinct2": 0}

    # ---- ROUGE ----
    rouge = Rouge()
    rouge_scores = []
    for ref, cand in zip(references, candidates):
        ref_seg = " ".join(jieba.cut(ref))
        cand_seg = " ".join(jieba.cut(cand))
        try:
            s = rouge.get_scores(cand_seg, ref_seg)
            rouge_scores.append(s[0]["rouge-1"]["f"])
        except:
            continue
    rouge1 = float(np.mean(rouge_scores)) if rouge_scores else 0.0

    # ---- BERTScore ----
    _, _, F = bert_score(candidates, references, lang="zh", verbose=False)
    bert_f1 = float(F.mean())

    # ---- Distinct ----
    all_chars = [list(c) for c in candidates]
    total_uni = sum(len(seq) for seq in all_chars)
    uniq_uni = len({c for seq in all_chars for c in seq})
    distinct1 = uniq_uni / total_uni if total_uni > 0 else 0.0

    bigrams = []
    for seq in all_chars:
        bigrams.extend([(seq[i], seq[i+1]) for i in range(len(seq)-1)])
    total_bi = len(bigrams)
    uniq_bi = len(set(bigrams))
    distinct2 = uniq_bi / total_bi if total_bi > 0 else 0.0

    return dict(
        rouge1=rouge1,
        bertscore_f1=bert_f1,
        distinct1=distinct1,
        distinct2=distinct2,
    )



def aggregate_over_seeds(seed_results: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    def _agg(key: str):
        vals = [r[key] for r in seed_results]
        return float(np.mean(vals)), float(np.std(vals))

    return {
        "rouge1": _agg("rouge1"),
        "bertscore_f1": _agg("bertscore_f1"),
        "distinct1": _agg("distinct1"),
        "distinct2": _agg("distinct2"),
    }


def eval_model_ccpc(model_name: str, path: str, seeds=(0, 1, 2)) -> Dict:
    model, tokenizer = load_model_and_tokenizer(path)

    seed_results = []
    for s in seeds:
        res = eval_ccpc_one_seed(model, tokenizer, seed=s, max_samples=300)
        seed_results.append(res)

    agg = aggregate_over_seeds(seed_results)
    print(f"\n[Aggregated] {model_name} -> {agg}")
    return {
        "model": model_name,
        "path": path,
        "seeds": list(seeds),
        "metrics": {
            "rouge1_mean": agg["rouge1"][0],
            "rouge1_std": agg["rouge1"][1],
            "bertscore_f1_mean": agg["bertscore_f1"][0],
            "bertscore_f1_std": agg["bertscore_f1"][1],
            "distinct1_mean": agg["distinct1"][0],
            "distinct1_std": agg["distinct1"][1],
            "distinct2_mean": agg["distinct2"][0],
            "distinct2_std": agg["distinct2"][1],
        },
    }


def main():
    global FAST_EVAL

    max_samples = FAST_SAMPLES if FAST_EVAL else NORMAL_SAMPLES

    results = []

    for name, path in MODEL_PATHS.items():
        print(f"\n====== Evaluating {name} ======\n")

        try:
            model, tokenizer = load_model_and_tokenizer(path)

            # seeds: fast_eval → 1 seed；normal → 3 seeds
            seeds = (0,) if FAST_EVAL else (0, 1, 2)

            seed_results = []
            for s in seeds:
                print(f"\n--- seed={s} ---")
                res = eval_ccpc_one_seed(model, tokenizer, s, max_samples=max_samples)
                seed_results.append(res)

            # 聚合
            agg = aggregate_over_seeds(seed_results)
            results.append({"model": name, "path": path, "agg": agg})

        except Exception as e:
            print(f"\n[Model Error: {name}] {e}")
            results.append({"model": name, "error": str(e)})
            continue

    with open("results_rouge_bertscore.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n[Done] All results written to results_rouge_bertscore.json\n")



if __name__ == "__main__":
    main()