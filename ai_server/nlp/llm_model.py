#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model
from utils.logger import setup_logger

# 设置日志
logger = setup_logger('llm_model')


class LLMModel:
    """
    大语言模型封装类
    """

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")

        # 模型配置
        self.model_name = config.get("llm.model_name", "deepseek-ai/deepseek-llm-7b-chat")
        self.model_path = config.get("llm.model_path", "./models/deepseek-llm-7b-chat")
        self.use_lora = config.get("llm.use_lora", True)
        self.lora_path = config.get("llm.lora_path", "./models/lora")

        # 加载模型
        self.load_model()

    def load_model(self):
        """
        加载模型
        """
        logger.info(f"加载模型: {self.model_name}")

        try:
            # 量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            # 添加缓存机制
            self.response_cache = {}

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path if os.path.exists(self.model_path) else self.model_name,
                trust_remote_code=True
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path if os.path.exists(self.model_path) else self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            # 如果使用LoRA且存在LoRA权重，加载LoRA权重
            if self.use_lora and os.path.exists(self.lora_path):
                logger.info(f"加载LoRA权重: {self.lora_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_path
                )

            logger.info("模型加载完成")

        except Exception as e:
            logger.error(f"加载模型时出错: {e}", exc_info=True)
            raise

    def generate(self, prompt, max_length=1024, temperature=0.7):
        """
        生成回答
        """
        try:
            # 检查缓存
            cache_key = f"{prompt}_{max_length}_{temperature}"
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]

            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            response = response[len(prompt):].strip()

            self.response_cache[cache_key] = response

            # 限制缓存大小
            if len(self.response_cache) > 100:  # 最多缓存100条记录
                # 删除最早的记录
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]

            return response
            # 移除原始提示

        except Exception as e:
            logger.error(f"生成回答时出错: {e}", exc_info=True)
            return "很抱歉，我现在无法回答这个问题。"

        except Exception as e:
            logger.error(f"生成回答时出错: {e}", exc_info=True)
            return "很抱歉，我现在无法回答这个问题。"

    def get_embedding(self, text):
        """
        获取文本的嵌入向量
        """
        try:
            # 编码输入
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # 获取嵌入
            with torch.no_grad():
                outputs = self.model.get_input_embeddings()(inputs.input_ids)

            # 使用平均池化获取句子嵌入
            mask = inputs.attention_mask.unsqueeze(-1)
            masked_embeddings = outputs * mask
            summed = torch.sum(masked_embeddings, dim=1)
            counted = torch.sum(mask, dim=1)
            embedding = summed / counted

            return embedding.cpu().numpy()

        except Exception as e:
            logger.error(f"获取嵌入向量时出错: {e}", exc_info=True)
            return None