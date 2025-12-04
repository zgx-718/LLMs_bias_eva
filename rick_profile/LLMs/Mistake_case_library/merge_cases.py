# -*- coding: utf-8 -*-
import json
import os

def merge_json_files(input_files, output_file, duplicates_log="duplicates.txt"):
    """
    合并多个JSON文件到一个JSON文件中，保证subject_id不重复。
    如果重复则输出重复的subject_id，并记录保存/未保存的bias_type。
    新数据会追加到output_file（不重写）。
    """
    merged_data = []
    existing_ids = {}  # {subject_id: bias_type}
    duplicates_info = []  # [(subject_id, saved_bias_type, new_bias_type)]

    # 如果输出文件已存在，先读取它的数据，避免重复
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                merged_data = json.load(f)
                for item in merged_data:
                    sid = item.get("subject_id")
                    btype = item.get("bias_type")
                    existing_ids[sid] = btype
            except json.JSONDecodeError:
                merged_data = []

    # 处理输入文件
    for file_path in input_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"文件 {file_path} 不是有效的JSON，已跳过。")
                continue

            for item in data:
                sid = item.get("subject_id")
                btype = item.get("bias_type")
                if sid in existing_ids:
                    duplicates_info.append((sid, existing_ids[sid], btype))
                else:
                    merged_data.append(item)
                    existing_ids[sid] = btype

    # 写回合并后的数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # 输出并记录重复ID和bias_type
    if duplicates_info:
        print("发现重复 subject_id 及其 bias_type 对比：")
        with open(duplicates_log, "a", encoding="utf-8") as f:
            for sid, saved_btype, new_btype in duplicates_info:
                line = f"subject_id: {sid} | 已保存 bias_type: {saved_btype} | 未保存 bias_type: {new_btype}"
                print(line)
                f.write(line + "\n")
    else:
        print("没有发现重复 subject_id。")


if __name__ == "__main__":
    # 示例：合并 input1.json 和 input2.json 到 merged.json
    input_files = [

    ]  # 需要合并的偏倚案例文件
    output_file = ""  # 最终案例库文件
    merge_json_files(input_files, output_file)
