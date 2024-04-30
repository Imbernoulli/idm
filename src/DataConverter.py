import os
import json

def process_json_data(data):
    # 这里定义你对JSON数据的处理逻辑
    # 例如，这里我仅仅是将原数据添加一个新的键值对
    data['processed'] = True
    return data

def main(source_folder, target_folder):
    # 创建目标文件夹如果它不存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        # 确保是文件而且是JSON文件
        if os.path.isfile(file_path) and filename.endswith('.json'):
            # 读取JSON数据
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # 处理JSON数据
            processed_data = process_json_data(data)

            # 保存处理后的数据到目标文件夹
            target_file_path = os.path.join(target_folder, filename)
            with open(target_file_path, 'w', encoding='utf-8') as file:
                json.dump(processed_data, file, indent=4)

if __name__ == "__main__":
    source_folder = 'path/to/source/folder'  # 源文件夹路径
    target_folder = 'path/to/target/folder'  # 目标文件夹路径
    main(source_folder, target_folder)