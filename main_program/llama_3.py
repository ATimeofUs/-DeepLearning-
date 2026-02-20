import torch
import flash_attn
import sys

from transformers import (
    AutoModelForCausalLM, # LLM 的模型类
    AutoTokenizer, # LLM 的 tokenizer 类
    BitsAndBytesConfig, # 4-bit 量化配置类
    TextStreamer, # 用于流式输出文本的类
)  


def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

def main(model_path: str):
    # 1. 配置 4-bit 量化 (保持不变)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        attn_implementation="flash_attention_2",  # 使用 Flash Attention 2
        low_cpu_mem_usage=True,
    )

    # --- 初始化流式处理器 ---
    # skip_prompt=True 表示不在终端重复显示你输入的问题
    # skip_special_tokens=True 过滤掉 <|end_of_text|> 等标签
    streamer = TextStreamer(
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=False
    )

    # 4. 推理示例
    while True:
        while True:
            print("输入：")
            user_input = sys.stdin.readline().strip()
            if user_input.lower() in ["exit", "quit"]:
                exit(0)

            print(f"提示词为: {user_input}，是否继续？(y/n)")
            confirm = sys.stdin.readline().strip().lower()
            if confirm in ["y", "yes"]:
                break
                    

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)            

        print("模型回答: ", end="", flush=True)  # 先打印一个前缀

        # 生成响应
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                streamer=streamer,  # 关键：传入 streamer
                max_new_tokens=1024,  # 建议用 max_new_tokens 代替 max_length
                do_sample=True,  # 使用 temperature 时必须设为 True
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,  # 建议加上，防止 Llama 陷入死循环
            )
            
        cost_cuda = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
        print(f"\n本次推理消耗显存: {cost_cuda:.2f} GB")

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 路径匹配你的 Arch Linux 系统路径
    model_path = "/home/ping/model/meta-llama-Llama-3.1-8B"
    main(model_path)

"""



"""