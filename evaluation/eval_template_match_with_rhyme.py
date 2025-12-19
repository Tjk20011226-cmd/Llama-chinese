# eval_template_match_with_rhyme.py
# Template-Match baseline + 押韵一致度

import json
import random
import re
from collections import defaultdict

from rouge_chinese import Rouge
from bert_score import score as bert_score
import numpy as np
import torch

from rhyme_metric import average_rhyme_score

DATA_PATH = "data/vaild_ccpc_以关键字生成诗词.json"
random.seed(42)


def split_sentences(text: str):
    text = text.replace(" ", "").strip()
    if not text:
        return []
    return [s for s in re.split(r"[，,。\.！!？\?；;]", text) if s]


def detect_form_from_poem(poem: str):
    sents = split_sentences(poem)
    n = len(sents)
    if n not in (4, 8):
        return None
    lens = [len(s) for s in sents]
    if n == 4:
        if all(l == 5 for l in lens):
            return "五言绝句"
        if all(l == 7 for l in lens):
            return "七言绝句"
    elif n == 8:
        if all(l == 5 for l in lens):
            return "五言律诗"
        if all(l == 7 for l in lens):
            return "七言律诗"
    return None


def detect_form(example):
    inst = example.get("instruction", "")
    ans = example.get("output", "")
    for form in ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]:
        if form in inst:
            return form
    return detect_form_from_poem(ans)


def check_format(poem: str, form: str) -> bool:
    sents = split_sentences(poem)
    n = len(sents)
    lens = [len(s) for s in sents]
    if form == "五言绝句":
        return n == 4 and all(l == 5 for l in lens)
    if form == "七言绝句":
        return n == 4 and all(l == 7 for l in lens)
    if form == "五言律诗":
        return n == 8 and all(l == 5 for l in lens)
    if form == "七言律诗":
        return n == 8 and all(l == 7 for l in lens)
    return False


def calc_distinct_1_2(poems):
    all_chars, all_bigrams = [], []
    for p in poems:
        chars = list(p.replace("\n", "").replace(" ", ""))
        if not chars:
            continue
        all_chars.extend(chars)
        all_bigrams.extend([chars[i] + chars[i+1] for i in range(len(chars)-1)])
    if not all_chars:
        return 0.0, 0.0
    distinct1 = len(set(all_chars)) / len(all_chars)
    distinct2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    return distinct1, distinct2


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    usable_examples = []
    form_buckets = defaultdict(list)

    for ex in data:
        form = detect_form(ex)
        if form is None:
            continue
        ex["form"] = form
        usable_examples.append(ex)
        form_buckets[form].append(ex["output"])

    print("可用样本数:", len(usable_examples))
    for form, poems in form_buckets.items():
        print(f"{form}: 池子大小 = {len(poems)}")

    rouge = Rouge()

    format_flags = []
    rouge_scores = []
    baseline_outputs = []
    refs_all = []

    MAX_BERT_SAMPLES = 500
    sampled_examples = random.sample(
        usable_examples,
        min(MAX_BERT_SAMPLES, len(usable_examples))
    )

    print("参与评估的样本数（ROUGE + Format）:", len(usable_examples))
    print("用于 BERTScore 的样本数:", len(sampled_examples))

    # ROUGE / Format / Distinct
    for ex in usable_examples:
        form = ex["form"]
        ref = ex["output"]
        pool = form_buckets[form]
        if not pool:
            continue

        cand = random.choice(pool)
        if len(pool) > 1 and cand == ref:
            for _ in range(3):
                tmp = random.choice(pool)
                if tmp != ref:
                    cand = tmp
                    break

        format_flags.append(1.0 if check_format(cand, form) else 0.0)

        cand_seg = " ".join(list(cand))
        ref_seg = " ".join(list(ref))
        try:
            s = rouge.get_scores(cand_seg, ref_seg)
            rouge_scores.append(s[0]["rouge-1"]["f"])
        except Exception:
            continue

        baseline_outputs.append(cand)
        refs_all.append(ref)

    # BERTScore（采样）
    cands_for_bert = []
    refs_for_bert = []
    for ex in sampled_examples:
        form = ex["form"]
        ref = ex["output"]
        pool = form_buckets[form]
        if not pool:
            continue
        cand = random.choice(pool)
        if len(pool) > 1 and cand == ref:
            for _ in range(3):
                tmp = random.choice(pool)
                if tmp != ref:
                    cand = tmp
                    break
        cands_for_bert.append(cand)
        refs_for_bert.append(ref)

    if cands_for_bert:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"计算 BERTScore，中间使用设备: {device}")
        _, _, F1 = bert_score(
            cands_for_bert,
            refs_for_bert,
            lang="zh",
            model_type="bert-base-chinese",
            device=device,
            batch_size=16
        )
        bert_f1 = float(F1.mean().item())
    else:
        bert_f1 = 0.0

    distinct1, distinct2 = calc_distinct_1_2(baseline_outputs)

    format_acc = float(np.mean(format_flags)) if format_flags else 0.0
    rouge1 = float(np.mean(rouge_scores)) if rouge_scores else 0.0

    # 押韵一致度：参考诗 + Template-Match
    ref_rhyme = average_rhyme_score(refs_all)
    tm_rhyme  = average_rhyme_score(baseline_outputs)

    print("\n===== Template-Match Baseline 结果（采样 + 批量 BERTScore）=====")
    print(f"Format Accuracy: {format_acc * 100:.2f}%")
    print(f"ROUGE-1 (F):    {rouge1:.4f}")
    print(f"BERTScore (F1): {bert_f1:.4f}")
    print(f"Distinct-1:     {distinct1:.4f}")
    print(f"Distinct-2:     {distinct2:.4f}")
    print("==== Rhyme Consistency (CCPC) ====")
    print(f"Reference poems      : {ref_rhyme:.4f}")
    print(f"Template-Match       : {tm_rhyme:.4f}")


if __name__ == "__main__":
    main()
