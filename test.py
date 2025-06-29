import transformers
import os

def find_qwen2_source():
    """直接查找Qwen2源码"""
    
    # 获取transformers基础路径
    base_path = os.path.dirname(transformers.__file__)
    
    # Qwen2模型路径
    qwen2_path = os.path.join(base_path, "models", "qwen2")
    
    print(f"Qwen2模型路径: {qwen2_path}")
    print(f"路径是否存在: {os.path.exists(qwen2_path)}")
    
    if os.path.exists(qwen2_path):
        print("\n=== Qwen2目录内容 ===")
        files = os.listdir(qwen2_path)
        for file in sorted(files):
            file_path = os.path.join(qwen2_path, file)
            size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
            print(f"  {file:<30} ({size} bytes)")
            
        # 检查关键文件
        key_files = [
            "modeling_qwen2.py",
            "configuration_qwen2.py", 
            "tokenization_qwen2.py"
        ]
        
        print("\n=== 关键文件检查 ===")
        for key_file in key_files:
            file_path = os.path.join(qwen2_path, key_file)
            exists = os.path.exists(file_path)
            print(f"  {key_file:<25} {'✅' if exists else '❌'}")
            
        return qwen2_path
    else:
        return None

# 执行查找
qwen2_path = find_qwen2_source()