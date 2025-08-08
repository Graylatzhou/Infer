# 项目简介

本项目主要实现了以下功能：

1. **适配 Qwen3 和 Bert Encoder**  
   - 支持 Qwen3 模型的推理输出处理。  
   - 集成 Bert Encoder，用于特征提取和语义处理。

2. **接入 GPT SoVITS**  
   - 使用 GPT SoVITS 将模型的文本输出转换为语音。  

---

## 功能特点

- **文本到语音转换**：通过 GPT SoVITS 实现模型输出的语音化。  
- **多语言支持**：支持中文、英文等多语言处理。  
- **高效推理**：优化了 Qwen3 和 Bert 的推理流程，提升了性能。

---

## 使用方法

1. 克隆项目：
   ```bash
   git clone https://github.com/Graylatzhou/Infer.git
   cd Infer