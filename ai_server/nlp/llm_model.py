#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

        # 在llm_model.py中添加教学模板
    def create_teaching_prompt(self, concept, detail_level="medium", tone="neutral"):
        """
        创建教学提示模板
        """
        detail_instructions = {
            "very_basic": "用极其简单的语言，只解释最基础的内容。使用生活中的类比。",
            "basic": "用简单语言解释主要内容，避免专业术语。",
            "medium": "平衡基础和进阶内容，使用少量专业术语并解释它们。",
            "advanced": "详细解释内容，包括相关理论和应用，使用专业术语。",
            "expert": "深入探讨概念的高级方面，包括边界情况和最新研究。"
        }

        tone_instructions = {
            "encouraging": "使用鼓励性的语言，强调进步和可能性。",
            "neutral": "使用客观、平实的语言解释概念。",
            "positive": "使用积极的语言，强调概念的有趣和实用方面。",
            "empathetic": "表现出理解学习困难，承认概念可能具有挑战性。"
        }

        prompt = f"""
        以下是一个教学对话。你是NAO助教，一位善于解释概念的教学助手。

        请解释"{concept}"概念。{detail_instructions.get(detail_level, "")}
        {tone_instructions.get(tone, "")}

        回答应结构清晰，包含以下部分：
        1. 简明定义
        2. 关键特性
        3. 应用场景或例子

        NAO助教:
        """

        return prompt