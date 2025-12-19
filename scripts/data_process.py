import json
import csv
import os
import glob
import pandas as pd
from tqdm import tqdm

def read_large_json(file_path):
    """逐行读取 JSON 文件"""
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

def process_and_save_to_csv(json_file, csv_file):
    """处理 JSON 文件并保存到 CSV"""
    print(json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
    		data = json.load(f)
    new_data = []
    for item in  tqdm(data):
    		combined_text = f"{item['dataType']} {item['title']} {item['content']}"
    		new_data.append({'text': combined_text})
    	# 创建 DataFrame
    df = pd.DataFrame(new_data)

    # 保存为 CSV 文件
    df.to_csv(csv_file, index=False, encoding='utf-8')

def convert_multiple_json_to_csv(json_files, output_dir):
    """将多个 JSON 文件转换为多个 CSV 文件"""
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录（如果不存在）

    for json_file in json_files:
        # 构建 CSV 文件名
        csv_file = os.path.join(output_dir, os.path.basename(json_file).replace('.json', '.csv'))
        process_and_save_to_csv(json_file, csv_file)

# 使用示例
DATA_CACHE_DIR = "data"
data_dir = os.path.join(DATA_CACHE_DIR, "wudao源文件")
json_files = sorted(glob.glob(os.path.join(data_dir, "*.json"))) # JSON 文件列表
output_directory = 'data/output_csvs'  # 输出目录
convert_multiple_json_to_csv(json_files, output_directory)
