import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 你要评估的四种体裁
FORMS = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]


def init_llm(model_path="二阶段全微调_全诗词数据_1epoch"):
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_answer(model, tokenizer, prompt):
    # 和你之前 eval / rerank 一致
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        ids = ids.to("cuda")
    L = len(ids)

    outputs = model.generate(
        input_ids=ids,
        max_new_tokens=128,
        do_sample=True,
        top_k=5,
        top_p=0.95,
        temperature=0.3,
        repetition_penalty=1.3,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    toks = outputs[0].tolist()[L:]
    try:
        i = toks.index(tokenizer.eos_token_id)
        toks = toks[:i]
    except ValueError:
        pass

    text = tokenizer.decode(toks)
    text = text[len(prompt):].replace(" ", "").strip()
    return text


def strict_check_v3(poem: str, form: str) -> bool:
    """
    适度严格版 Strict-Format：
    - 不含明显 meta 词
    - 不含英文/数字
    - 去掉标点后，长度必须刚好等于 n_line * n_char
    - 切成行后，每个字符必须是汉字
    """

    # ---- 轻量 meta 过滤（内容洁净）----
    meta_bad = ["回答", "如下", "以下", "生成的", "请欣赏", "上述", "回复"]
    if any(m in poem for m in meta_bad):
        return False

    # 禁止英文、数字
    if re.search(r"[A-Za-z0-9]", poem):
        return False

    # ---- 结构参数 ----
    if form == "五言绝句":
        n_line, n_char = 4, 5
    elif form == "七言绝句":
        n_line, n_char = 4, 7
    elif form == "五言律诗":
        n_line, n_char = 8, 5
    elif form == "七言律诗":
        n_line, n_char = 8, 7
    else:
        return False

    # 去掉常见标点
    clean = re.sub(r"[，。？！,.!?、\s]", "", poem)

    # 要求长度“刚好”等于应有的总字数（比原指标严格）
    total = n_line * n_char
    if len(clean) != total:
        return False

    # 按字数硬切行
    lines = [clean[i * n_char:(i + 1) * n_char] for i in range(n_line)]

    # 每个字符必须是汉字
    for l in lines:
        if not l:  # 防御
            return False
        if not all("\u4e00" <= c <= "\u9fff" for c in l):
            return False

    return True


def strict_baseline(model, tokenizer, num_samples=100):
    correct = 0
    for _ in range(num_samples):
        form = random.choice(FORMS)
        prompt = f"写一首{form}。回答："
        poem = generate_answer(model, tokenizer, prompt)
        if strict_check_v3(poem, form):
            correct += 1
    acc = correct / num_samples * 100
    print(f"[Strict Baseline] Strict-Format Accuracy: {acc:.2f}%")
    return acc


def strict_rerank(model, tokenizer, num_samples=100, n_candidates=3):
    correct = 0
    for _ in range(num_samples):
        form = random.choice(FORMS)
        prompt = f"写一首{form}。回答："

        ok = False
        for i in range(n_candidates):
            poem = generate_answer(model, tokenizer, prompt)
            if strict_check_v3(poem, form):
                ok = True
                break

        if ok:
            correct += 1

    acc = correct / num_samples * 100
    print(f"[Strict Rerank] Strict-Format Accuracy (n={n_candidates}): {acc:.2f}%")
    return acc


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    model, tokenizer = init_llm("二阶段全微调_全诗词数据_1epoch")

    strict_baseline(model, tokenizer, num_samples=100)
    strict_rerank(model, tokenizer, num_samples=100, n_candidates=3)
