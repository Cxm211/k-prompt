import os
import shutil
import json
import random
# 定义要创建的主文件夹前缀
prefixes = ["sstubs", "bugsinpy", "tfix", "xcodeeval"]

# 定义大小和编号
sizes = [1, 8, 16, 32, 100, 300, 500, 700, 900]
numbers = [1, 2, 3]

# 定义要复制的文件
source_files = ["validation.jsonl"]

def create_folders(prefixes, sizes, numbers):
    """
    创建符合特定规则的文件夹。
    """
    for prefix in prefixes:
        for size in sizes:
            for number in numbers:
                # 拼接新的文件夹名称
                folder_name = f"{prefix}_{size}_{number}"
                # 创建目录
                os.makedirs(folder_name, exist_ok=True)
                print(f"Created folder: {folder_name}")

def sample(prefix):
    with open("validation.jsonl", "r") as validf:
        lines = validf.readlines()
    valid_output = []
    for line in lines:
        data = json.loads(line)
        if len(data['problem'].replace("\n","").replace("\r","").replace("\t","")) < 1000 and len(data['fixed'].replace("\n","").replace("\r","").replace("\t","")) < 1000:
            valid_output.append(line)
        valid1 = random.sample(valid_output, min(100, len(valid_output)))
        valid2 = random.sample(valid_output, min(100, len(valid_output)))
        valid3 = random.sample(valid_output, min(100, len(valid_output)))
            

    with open("test.jsonl", "r") as testf:
        lines = testf.readlines()
    test_output = []
    for line in lines:
        data = json.loads(line)
        if len(data['problem'].replace("\n","").replace("\r","").replace("\t","")) < 1000 and len(data['fixed'].replace("\n","").replace("\r","").replace("\t","")) < 1000:
            test_output.append(line)
        test1 = random.sample(test_output, min(500, len(test_output)))
        test2 = random.sample(test_output, min(500, len(test_output)))
        test3 = random.sample(test_output, min(500, len(test_output)))

    for folder_name in os.listdir():
        # 检查是否是目录且名称以给定前缀开头，并且匹配特定编号
        if os.path.isdir(folder_name) and folder_name.startswith(prefix):
            
            parts = folder_name.split('_')
            if len(parts) == 3 and parts[2].isdigit():
                if int(parts[2]) == 1:
                    destination_file = os.path.join(folder_name, "validation.jsonl")
                    with open(destination_file, "w") as f:
                        f.writelines(valid1)
                    print(f"Sampled {len(valid1)} items into {destination_file}")
                    destination_file = os.path.join(folder_name, "test.jsonl")
                    with open(destination_file, "w") as f:
                        f.writelines(test1)
                    print(f"Sampled {len(test1)} items into {destination_file}")
                elif int(parts[2]) == 2:
                    destination_file = os.path.join(folder_name, "validation.jsonl")
                    with open(destination_file, "w") as f:
                        f.writelines(valid2)
                    print(f"Sampled {len(valid2)} items into {destination_file}")
                    destination_file = os.path.join(folder_name, "test.jsonl")
                    with open(destination_file, "w") as f:
                        f.writelines(test2)
                    print(f"Sampled {len(test2)} items into {destination_file}")
                elif int(parts[2]) == 3:
                    destination_file = os.path.join(folder_name, "validation.jsonl")
                    with open(destination_file, "w") as f:
                        f.writelines(valid3)
                    print(f"Sampled {len(valid3)} items into {destination_file}")
                    destination_file = os.path.join(folder_name, "test.jsonl")
                    with open(destination_file, "w") as f:
                        f.writelines(test3)
                    print(f"Sampled {len(test3)} items into {destination_file}")

sample("sstubs")

# def copy_files_to_folders(prefix, numbers, source_files):
#     """
#     将指定的文件复制到所有符合给定前缀和特定编号的文件夹中。
#     """
#     # 列出当前目录下所有文件夹
#     for folder_name in os.listdir():
#         # 检查是否是目录且名称以给定前缀开头，并且匹配特定编号
#         if os.path.isdir(folder_name) and folder_name.startswith(prefix):
#             # 提取文件夹名称中的编号部分
#             parts = folder_name.split('_')
#             if len(parts) == 3 and parts[2].isdigit():
#                 folder_number = int(parts[2])
#                 if folder_number in numbers:
#                     # 将文件复制到匹配的文件夹中
#                     for source_file in source_files:
#                         # 检查源文件是否存在
#                         if os.path.exists(source_file):
#                             # 构造目标路径
#                             destination = os.path.join(folder_name, source_file)
#                             # 复制文件
#                             shutil.copy(source_file, destination)
#                             print(f"Copied {source_file} to {folder_name}")

# 调用函数创建文件夹
# create_folders(prefixes, sizes, numbers)

# 调用函数复制文件（以bugsinpy为例，复制到特定编号为1和2的文件夹）
# copy_files_to_folders("bugsinpy", [1,2,3], source_files)

# print("文件夹创建完毕，文件复制完毕。")


import os
import random
import json

def sample_data_from_jsonl(prefix, numbers):
    """
    从当前目录下的 test.jsonl 文件中随机抽取数据，并将抽取的数据保存到指定前缀和编号的文件夹中。
    每次抽取的数据条数根据文件夹名称中的 size 决定。
    """
    # 检查 test.jsonl 文件是否存在
    jsonl_filename = "train.jsonl"
    if not os.path.exists(jsonl_filename):
        print(f"File {jsonl_filename} not found!")
        return

    # 读取 test.jsonl 文件中的所有数据
    with open(jsonl_filename, "r") as f:
        lines = f.readlines()

    if not lines:
        print("No data found in train.jsonl!")
        return

    # 列出当前目录下所有文件夹
    for folder_name in os.listdir():
        # 检查是否是目录且名称以给定前缀开头
        if os.path.isdir(folder_name) and folder_name.startswith(prefix):
            # 提取文件夹名称中的编号部分
            parts = folder_name.split('_')
            if len(parts) == 3 and parts[2].isdigit():
                folder_number = int(parts[2])
                if folder_number in numbers:
                    # 获取文件夹的大小部分（即抽样数据的数量）
                    size = int(parts[1])
                    # 随机抽取指定数量的样本
                    outputs = []
                    for line in lines:
                        data = json.loads(line)
                        if len(data['problem'].replace("\n","").replace("\r","").replace("\t","")) < 1000 and len(data['fixed'].replace("\n","").replace("\r","").replace("\t","")) < 1000:
                            outputs.append(line)
                    sampled_data = random.sample(outputs, min(size, len(lines)))
                    
                    # 构造输出文件路径
                    destination_file = os.path.join(folder_name, "train.jsonl")
                    
                    # 将抽样数据写入目标文件夹中的 test.jsonl
                    with open(destination_file, "w") as f:
                        f.writelines(sampled_data)
                    print(f"Sampled {len(sampled_data)} items into {destination_file}")

sample_data_from_jsonl("sstubs", [1, 2, 3])

# print("数据采样完毕并存储到目标文件夹中。")
