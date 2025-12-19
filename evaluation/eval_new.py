# eval_new.py
# 目的：在不改动原 eval.py 的前提下，
# 单独实现：
#  - 多随机种子 Format Accuracy [R1-7]
#  - CCPC 上的 ROUGE-1 + BERTScore + Distinct-1/2 [R3-1, R1-7]

import json
import random
import numpy as np
import torch
import jieba
from tqdm import tqdm

from rouge_chinese import Rouge           # [R3-1] 用于 ROUGE-1
from bert_score import score as bert_score  # [R3-1] 用于 BERTScore 语义相似度

# 注意：这里导入你原来的 eval.py，为避免和内置 eval 函数冲突，改名为 eval_orig
import eval as eval_orig                  # eval.py 必须和本文件在同一目录


def set_random_seed(seed: int):
    """统一设置随机种子  [R1-7]"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_acc_multi(model, enc, seeds=(0, 1, 2)):
    """
    多随机种子下的格式准确率评测  [R1-7]

    直接复用你原 eval.py 里的 geshi(model, enc)，
    前提：原来的 geshi 最后已经被你改成 return 百分比。
    如果你还没改，只打印不 return，可以先在 eval.py 里给 geshi 最后一行加上：
        return correct_ratio
    """
    results = []
    for seed in seeds:
        set_random_seed(seed)
        acc = eval_orig.geshi(model, enc)  # 复用原来的逻辑
        print(f"[FormatAcc] seed={seed}, acc={acc:.2f}")
        results.append(acc)

    if not results:
        print("[FormatAcc] 无有效结果")
        return 0.0, 0.0

    mean = float(np.mean(results))
    std = float(np.std(results))
    print(f"[FormatAcc] mean={mean:.2f}, std={std:.2f}")
    return mean, std


def eval_ccpc_generation_multi(model, enc, seeds=(0, 1, 2), max_samples: int = 300):
    """
    在 CCPC 上评估生成质量  [R3-1, R1-7]
    - ROUGE-1 F
    - BERTScore F1
    - Distinct-1 / Distinct-2（Li et al. 2016 标准定义）
    多个种子，返回各指标 mean/std。
    """

    # 路径与原 eval.py 中一致，如有不同，请改成你真实使用的 json 路径
    with open("data/vaild_ccpc_以关键字生成诗词.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_data = val_data[:max_samples]

    all_seed_results = []

    for seed in seeds:
        set_random_seed(seed)
        references = []
        candidates = []

        for example in tqdm(val_data, desc=f"CCPC seed={seed}"):
            # 这里字段名基于你原 eval.py 的实现：
            #   example["instruction"], example["prompt"], example["output"]
            instruction = example["instruction"]
            prompt = example["prompt"]
            answer = example["output"]

            full_prompt = instruction + "\n" + prompt + "\n回答:"
            inputs = enc(full_prompt, return_tensors="pt", add_special_tokens=False).input_ids
            len_prompt = len(inputs)
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            # 生成参数严格参考你原 eval.py / geshi 的设置
            generate_input = {
                "input_ids": inputs,
                "max_new_tokens": 512,
                "do_sample": True,
                "top_k": 5,
                "top_p": 0.95,
                "temperature": 0.3,
                "repetition_penalty": 1.3,
                "eos_token_id": enc.eos_token_id,
                "bos_token_id": enc.bos_token_id,
                "pad_token_id": enc.pad_token_id,
            }
            outputs = model.generate(**generate_input)
            pt_tokens = outputs[0].tolist()
            pt_tokens = pt_tokens[len_prompt:]
            try:
                index = pt_tokens.index(enc.eos_token_id)
            except ValueError:
                # 没有遇到 eos，跳过这一条
                continue
            chunk = pt_tokens[:index]
            output_text = enc.decode(chunk)
            # 去掉 prompt 本身
            output_text = output_text[len(full_prompt):].strip()

            references.append(answer.strip())
            candidates.append(output_text)

        if not candidates:
            print(f"[CCPC] seed={seed} 无有效样本，跳过")
            continue

        # === ROUGE-1 F ===  [R3-1]
        rouge = Rouge()
        rouge_scores = []
        for ref, cand in zip(references, candidates):
            ref_seg = " ".join(jieba.cut(ref))
            cand_seg = " ".join(jieba.cut(cand))
            try:
                s = rouge.get_scores(cand_seg, ref_seg)
                rouge_scores.append(s[0]["rouge-1"]["f"])
            except Exception:
                continue
        rouge1 = float(np.mean(rouge_scores)) if rouge_scores else 0.0

        # === BERTScore F1 ===  [R3-1] 短诗语义相似度
        # 不做分词，直接输入中文句子
        P, R, F = bert_score(candidates, references, lang="zh", verbose=False)
        bert_f1 = float(F.mean())

        # === Distinct-1 / 2（标准定义：输出内部多样性） [R3-1] ===
        # 这里按“字”计算 n-gram，更严格地反映用字多样性
        all_chars = [list(cand.replace(" ", "")) for cand in candidates]
        # unigram
        total_unigrams = sum(len(seq) for seq in all_chars)
        uniq_unigrams = len({ch for seq in all_chars for ch in seq})
        distinct1 = uniq_unigrams / total_unigrams if total_unigrams > 0 else 0.0
        # bigram
        bigrams = []
        for seq in all_chars:
            bigrams.extend([(seq[i], seq[i + 1]) for i in range(len(seq) - 1)])
        total_bigrams = len(bigrams)
        uniq_bigrams = len(set(bigrams))
        distinct2 = uniq_bigrams / total_bigrams if total_bigrams > 0 else 0.0

        result = {
            "seed": seed,
            "rouge1": rouge1,
            "bertscore_f1": bert_f1,
            "distinct1": distinct1,
            "distinct2": distinct2,
        }
        print(f"[CCPC] seed={seed} -> {result}")
        all_seed_results.append(result)

    # 汇总 mean / std  [R1-7]
    if not all_seed_results:
        print("[CCPC] 没有任何有效结果")
        return None

    def _agg(key):
        vals = [r[key] for r in all_seed_results]
        return float(np.mean(vals)), float(np.std(vals))

    agg = {
        "rouge1": _agg("rouge1"),
        "bertscore_f1": _agg("bertscore_f1"),
        "distinct1": _agg("distinct1"),
        "distinct2": _agg("distinct2"),
    }
    print("[CCPC] aggregated (mean, std):", agg)
    return agg


def main():
    # 从原 eval.py 复用模型加载逻辑
    model, enc = eval_orig.int_llm()

    # 1) 多种子格式准确率 [R1-7]
    format_acc_multi(model, enc, seeds=(0, 1, 2))

    # 2) CCPC 生成质量 [R3-1, R1-7]
    eval_ccpc_generation_multi(model, enc, seeds=(0, 1, 2), max_samples=300)


if __name__ == "__main__":
    main()
