import json
import os
import traceback
import importlib
import torch

MODEL_PATHS = {
    "alpha_0_variant": "/home/chengcheng/data/Llama-Chinese/全微调_全诗词_2epoch_修改辅助损失",
    "alpha_0.1": "/home/chengcheng/data/Llama-Chinese/全微调_全诗词_2epoch_损失0.1",
    "alpha_0.2": "/home/chengcheng/data/Llama-Chinese/全微调_全诗词_2epoch_损失0.2",
    "alpha_0.3": "/home/chengcheng/data/Llama-Chinese/全微调_诗词_2epoch_消融辅助损失0.3",
    "alpha_0.4": "/home/chengcheng/data/Llama-Chinese/全微调_全诗词_2epoch_损失0.4",
    "alpha_0.5": "/home/chengcheng/data/Llama-Chinese/全微调_全诗词_2epoch_损失0.5",

    "with_identifier": "/home/chengcheng/data/Llama-Chinese/二阶段全微调_全诗词数据_1epoch",
    "without_identifier": "/home/chengcheng/data/Llama-Chinese/全微调_诗词_ccpm_2epoch_标识符消融",
}

# ===== Step 2: 引入 eval_new.py =====
import eval_new

# 修改 eval_orig.int_llm() 的模型路径
import eval as eval_orig

def replace_model_path(new_path):
    """
    动态替换 eval_orig.int_llm() 内的模型路径
    """
    def new_int_llm():
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        model = torch.load  # dummy just for placeholder

        model = eval_orig.AutoModelForCausalLM.from_pretrained(
            new_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            load_in_8bit=False,
            trust_remote_code=True,
            use_flash_attention_2=False
        ).eval()

        tokenizer = eval_orig.AutoTokenizer.from_pretrained(new_path, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    eval_orig.int_llm = new_int_llm


# ===== Step 3: 依次评估每个模型 =====
def main():
    results = {}

    for name, path in MODEL_PATHS.items():
        print(f"\n==============================")
        print(f"Running model: {name}")
        print(f"Path: {path}")
        print(f"==============================\n")

        if not os.path.exists(path):
            print(f"[ERROR] 路径不存在，跳过: {path}")
            continue

        try:
            # 替换模型路径
            replace_model_path(path)

            # 调用 eval_new.main() 执行评估
            out = eval_new.main()

            # 保存结果
            results[name] = "OK"

        except Exception as e:
            print(f"[ERROR] 模型 {name} 评估失败，跳过")
            traceback.print_exc()
            results[name] = "FAILED"

        torch.cuda.empty_cache()

    # 保存结果
    with open("all_eval_status.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n===== 全部模型评估完成 =====")
    print("结果保存在 all_eval_status.json")


if __name__ == "__main__":
    main()
