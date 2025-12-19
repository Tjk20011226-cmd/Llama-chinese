import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypinyin import pinyin, Style


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


def generate_answer(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    len_prompt = len(inputs)

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

    try:
        index = pt_tokens.index(tokenizer.eos_token_id)
        pt_tokens = pt_tokens[:index]
    except ValueError:
        pass

    output = tokenizer.decode(pt_tokens)
    output_ = output[len(prompt):].replace(" ", "")
    return output_


def get_final_pinyin(ch: str):
    py = pinyin(ch, style=Style.FINALS, strict=False)
    if not py or not py[0]:
        return None
    return py[0][0]


def get_tone_type(ch: str):
    py = pinyin(ch, style=Style.TONE3, strict=False)
    if not py or not py[0]:
        return None
    s = py[0][0]
    tone = None
    for c in reversed(s):
        if c.isdigit():
            tone = int(c)
            break
    if tone is None:
        return None
    return "平" if tone in (1, 2) else "仄"


def analyze_poem(poem: str, target_form: str):
    """
    使用和 Format Accuracy 相同的句子切分方式：
    - 用 ，。？！ 分句
    - 根据 target_form 取前 4 句或前 8 句做韵律分析
    """
    sentences = [s for s in re.split("[，。？！]", poem) if s.strip() != ""]
    if target_form in ["五言绝句", "七言绝句"]:
        if len(sentences) < 4:
            return None
        lines = sentences[:4]
    elif target_form in ["五言律诗", "七言律诗"]:
        if len(sentences) < 8:
            return None
        lines = sentences[:8]
    else:
        return None

    # 偶数句作为押韵句（第 2、4、6、8 句）
    rhyme_lines_idx = [i for i in range(len(lines)) if (i + 1) % 2 == 0]

    # ---- 押韵一致性：看偶数句句尾韵母是否一致 ----
    finals = []
    for i in rhyme_lines_idx:
        last_char = lines[i][-1]
        f = get_final_pinyin(last_char)
        if f:
            finals.append(f)

    rhyme_consistency = 0.0
    if len(finals) >= 2:
        from collections import Counter
        cnt = Counter(finals)
        most_common = cnt.most_common(1)[0][1]
        rhyme_consistency = most_common / len(finals)

    # ---- 平仄统计：偶数句句末是平还是仄 ----
    ping_count, ze_count, total_tone = 0, 0, 0
    for i in rhyme_lines_idx:
        last_char = lines[i][-1]
        t = get_tone_type(last_char)
        if t is None:
            continue
        total_tone += 1
        if t == "平":
            ping_count += 1
        else:
            ze_count += 1

    ping_ratio = ping_count / total_tone if total_tone > 0 else 0.0
    ze_ratio = ze_count / total_tone if total_tone > 0 else 0.0

    return {
        "num_lines": len(lines),
        "rhyme_consistency": rhyme_consistency,
        "ping_ratio_even_ends": ping_ratio,
        "ze_ratio_even_ends": ze_ratio,
    }


def prosody_eval(model, tokenizer, num_samples=50):
    forms = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]
    stats = []

    for _ in range(num_samples):
        geshi_choice = random.choice(forms)
        prompt = f"写一首诗，要求：{geshi_choice}。回答："
        poem = generate_answer(model, tokenizer, prompt)
        res = analyze_poem(poem, geshi_choice)
        if res is not None:
            stats.append(res)

    if not stats:
        print("没有得到有效的绝句 / 律诗样本。")
        return

    avg_rhyme = sum(s["rhyme_consistency"] for s in stats) / len(stats)
    avg_ping = sum(s["ping_ratio_even_ends"] for s in stats) / len(stats)
    avg_ze = sum(s["ze_ratio_even_ends"] for s in stats) / len(stats)

    print(f"样本数量: {len(stats)}")
    print(f"平均押韵一致性（偶数句同韵母比例）: {avg_rhyme:.3f}")
    print(f"偶数句句尾平声比例: {avg_ping:.3f}")
    print(f"偶数句句尾仄声比例: {avg_ze:.3f}")


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    model, tokenizer = init_llm("二阶段全微调_全诗词数据_1epoch")
    prosody_eval(model, tokenizer, num_samples=50)

