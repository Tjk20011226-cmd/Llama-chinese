import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 1. 模型加载（直接复用 eval.py 里的设置） =====
def init_llm(model_path="二阶段全微调_全诗词数据_1epoch"):
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        trust_remote_code=True,
        use_flash_attention_2=False,
    )
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ===== 2. 生成一首诗：复用你 eval.py 里的写法 =====
def generate_answer(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    len_prompt = len(inputs)  # 注意：你的原始代码也是这样写的

    generate_input = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": 5,
        "top_p": 0.95,
        "temperature": 0.3,
        "repetition_penalty": 1.3,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    outputs = model.generate(**generate_input)
    pt_tokens = outputs[0].tolist()
    pt_tokens = pt_tokens[len_prompt:]

    # 过滤异常 token（和你原代码一致）
    to_remove = {55296, 55297, 55298, 55299}
    pt_tokens = [x for x in pt_tokens if x not in to_remove]

    try:
        index = pt_tokens.index(tokenizer.eos_token_id)
        pt_tokens = pt_tokens[:index]
    except ValueError:
        # 没有 EOS，就全部保留
        pass

    output = tokenizer.decode(pt_tokens)
    # 你的原代码是 output[len(prompt):]，保持一致以避免指标不一致
    output_ = output[len(prompt):]
    output_ = output_.replace(" ", "")
    return output_  # 只返回模型的“回答”部分（纯诗句）


# ===== 3. 体裁识别：直接抽出你在 geshi() 里的逻辑 =====
def detect_format(poem: str) -> str:
    """
    输入：模型生成的诗（不含 prompt）
    输出：识别到的体裁（五言绝句 / 七言绝句 / 五言律诗 / 七言律诗 / ""）
    """
    sentences = [s for s in re.split("[，。？！]", poem) if s != ""]
    ticai = ""
    if len(sentences) == 4:
        if all(len(s) == 5 for s in sentences):
            ticai = "五言绝句"
        elif all(len(s) == 7 for s in sentences):
            ticai = "七言绝句"
    elif len(sentences) == 8:
        if all(len(s) == 5 for s in sentences):
            ticai = "五言律诗"
        elif all(len(s) == 7 for s in sentences):
            ticai = "七言律诗"
    return ticai


# ===== 4. Baseline：你的原始 Format Accuracy 实验 =====
def geshi_baseline(model, tokenizer, num_samples=100):
    forms = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]
    scores = []

    for _ in range(num_samples):
        geshi_choice = random.choice(forms)
        prompt = f"写一首诗，要求：{geshi_choice}。回答："
        poem = generate_answer(model, tokenizer, prompt)
        pred_form = detect_format(poem)
        correct = 1 if pred_form == geshi_choice else 0
        scores.append(correct)

    acc = 100 * sum(scores) / len(scores)
    print(f"[Baseline] 格式准确度: {acc:.2f}%")


# ===== 5. Rerank 解码：多生成几首，优先选择格式合法的 =====
def geshi_rerank(model, tokenizer, num_samples=100, n_candidates=4):
    forms = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]
    scores = []

    for _ in range(num_samples):
        geshi_choice = random.choice(forms)
        prompt = f"写一首诗，要求：{geshi_choice}。回答："

        chosen_form = ""
        # 生成多候选
        for i in range(n_candidates):
            poem = generate_answer(model, tokenizer, prompt)
            pred_form = detect_format(poem)
            if pred_form == geshi_choice:
                chosen_form = pred_form
                break  # 一旦生成到合法格式，就选它
            if i == 0:
                chosen_form = pred_form  # 全部不合法时，回退到第一首

        correct = 1 if chosen_form == geshi_choice else 0
        scores.append(correct)

    acc = 100 * sum(scores) / len(scores)
    print(f"[Rerank]  格式准确度 (候选数={n_candidates}): {acc:.2f}%")


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    model, tokenizer = init_llm("二阶段全微调_全诗词数据_1epoch")
    geshi_baseline(model, tokenizer, num_samples=100)
    geshi_rerank(model, tokenizer, num_samples=100, n_candidates=3)
