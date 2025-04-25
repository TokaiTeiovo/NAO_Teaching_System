# knowledge_extraction/llm_knowledge_extractor.py
import os
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from tqdm import tqdm

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_knowledge_extractor')


class LLMKnowledgeExtractor:
    """
    使用本地大模型提取知识图谱
    """

    def __init__(self, model_path=None):
        """
        初始化大模型提取器

        参数:
            model_path: 模型路径，如果为None则使用默认的DeepSeek模型
        """
        self.model_path = model_path or "deepseek-ai/deepseek-llm-7b-chat"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载模型
        logger.info(f"加载大模型: {self.model_path}")
        print(f"加载大模型: {self.model_path}")
        self._load_model()

    def _load_model(self):
        """加载大模型"""
        try:
            # 检查模型路径是文件夹还是模型名称
            is_local_path = os.path.exists(self.model_path)
            logger.info(f"使用{'本地' if is_local_path else '远程'}模型: {self.model_path}")
            print(f"使用{'本地' if is_local_path else '远程'}模型: {self.model_path}")

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # 内存优化设置
            # 使用8位量化代替4位量化，降低内存需求
            # 注意：这会略微降低模型质量，但可以缓解内存不足问题
            try:
                # 首先尝试加载4位量化模型，但允许CPU卸载
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_enable_fp32_cpu_offload=True  # 启用CPU卸载
                )# 创建自定义设备映射
                # 这将把一些模块放在CPU上
                print("尝试加载4位量化模型，部分模块卸载到CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",  # 自动决定分配
                    max_memory={0: "8GiB", "cpu": "32GiB"},  # 限制GPU使用的内存，可以根据实际情况调整
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                print(f"4位量化加载失败: {e}")
                print("尝试使用8位量化...")

                # 如果4位量化失败，尝试8位量化
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,  # 使用8位量化
                    llm_int8_enable_fp32_cpu_offload=True
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    max_memory={0: "8GiB", "cpu": "32GiB"},
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

            logger.info("模型加载完成")
            print("模型加载完成")

        except Exception as e:

            logger.error(f"加载模型时出错: {e}")

            print(f"加载模型时出错: {e}")

        # 如果量化模型加载失败，尝试加载到CPU上

        try:

            print("尝试将模型完全加载到CPU...")

            self.model = AutoModelForCausalLM.from_pretrained(

                self.model_path,

                device_map="cpu",

                torch_dtype=torch.float16,

                trust_remote_code=True,

                low_cpu_mem_usage=True

            )

            logger.info("模型已加载到CPU")

            print("模型已加载到CPU")

        except Exception as cpu_e:

            logger.error(f"CPU加载也失败: {cpu_e}")

            print(f"CPU加载也失败: {cpu_e}")

            raise

    def _generate_text(self, prompt, max_length=2048, temperature=0.7):
        """生成文本"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 移除原始提示
            prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            response = response[len(prompt_text):].strip()

            return response
        except Exception as e:
            logger.error(f"生成文本时出错: {e}")
            print(f"生成文本时出错: {e}")
            return ""

    def extract_knowledge_from_text(self, text, chapter_title=""):
        """从文本中提取知识点"""
        try:
            # 构建提示
            prompt = f"""
你是一个知识图谱提取专家。请从下面的编译原理教材文本中提取关键概念及其定义。
请按照以下JSON格式输出提取的知识点:
[
  {{
    "concept": "概念名称",
    "definition": "概念定义",
    "chapter": "{chapter_title}",
    "importance": 4,
    "difficulty": 3
  }}
]"""

            # 由于模型的上下文长度限制，我们需要限制输入文本的长度
            max_text_length = 4000  # 根据您的模型调整这个值
            if len(text) > max_text_length:
                text = text[:max_text_length]

            # 添加文本到提示
            full_prompt = prompt + "\n\n" + text

            # 生成回答
            response = self._generate_text(full_prompt)

            # 提取JSON部分
            try:
                # 尝试匹配JSON数组
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # 尝试匹配任何可能的JSON结构
                    json_str = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_str:
                        json_str = "[" + json_str.group(0) + "]"
                    else:
                        # 没有找到JSON，返回空列表
                        print(f"未能从响应中提取JSON结构: {response[:100]}...")
                        return []

                # 尝试解析JSON
                try:
                    knowledge_points = json.loads(json_str)
                    return knowledge_points
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    print(f"尝试修复JSON...")

                    # 尝试修复常见的JSON错误
                    fixed_json = json_str.replace("'", '"')  # 替换单引号为双引号
                    fixed_json = re.sub(r',\s*\]', ']', fixed_json)  # 移除数组末尾多余的逗号

                    try:
                        knowledge_points = json.loads(fixed_json)
                        print("JSON修复成功!")
                        return knowledge_points
                    except:
                        print("JSON修复失败，返回空列表")
                        return []

            except Exception as e:
                logger.error(f"处理JSON时出错: {e}")
                print(f"处理JSON时出错: {e}")
                return []

        except Exception as e:
            logger.error(f"提取关系时出错: {e}")
            print(f"提取关系时出错: {e}")
            return []

    def process_chapters(self, chapters):
        """
        处理多个章节

        参数:
            chapters: 章节字典，格式为 {chapter_title: chapter_text}

        返回:
            知识点列表和关系列表
        """
        all_knowledge_points = []

        # 从每个章节提取知识点
        for chapter_title, chapter_info in tqdm(chapters.items(), desc="处理章节"):
            chapter_text = chapter_info["text"]

            # 跳过太短的章节
            if len(chapter_text) < 100:
                print(f"跳过过短的章节: {chapter_title}")
                continue

            print(f"正在从章节 '{chapter_title}' 提取知识点...")

            # 提取知识点
            knowledge_points = self.extract_knowledge_from_text(chapter_text, chapter_title)

            if knowledge_points:
                all_knowledge_points.extend(knowledge_points)
                print(f"从章节 '{chapter_title}' 提取了 {len(knowledge_points)} 个知识点")
            else:
                print(f"未能从章节 '{chapter_title}' 提取知识点")

        # 提取关系
        print("\n正在提取概念间的关系...")
        relationships = self.extract_relationships_from_knowledge(all_knowledge_points)

        print(f"总共提取了 {len(all_knowledge_points)} 个知识点和 {len(relationships)} 个关系")

        return all_knowledge_points, relationships

    def create_knowledge_graph(self, knowledge_points, relationships, output_path):
        """
        创建知识图谱并保存为JSON

        参数:
            knowledge_points: 知识点列表
            relationships: 关系列表
            output_path: 输出路径

        返回:
            是否成功
        """
        try:
            # 创建节点
            nodes = []
            for kp in knowledge_points:
                node = {
                    "id": kp["concept"],
                    "name": kp["concept"],
                    "type": "Concept",
                    "definition": kp.get("definition", ""),
                    "chapter": kp.get("chapter", ""),
                    "importance": kp.get("importance", 3),
                    "difficulty": kp.get("difficulty", 3)
                }
                nodes.append(node)

            # 创建链接
            links = []
            for rel in relationships:
                link = {
                    "source": rel["source"],
                    "target": rel["target"],
                    "type": rel["relation"],
                    "strength": rel.get("strength", 0.5)
                }
                links.append(link)

            # 创建知识图谱
            graph = {
                "nodes": nodes,
                "links": links
            }

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)

            print(f"知识图谱已保存到: {output_path}")
            print(f"包含 {len(nodes)} 个节点和 {len(links)} 个链接")

            return True

        except Exception as e:
            logger.error(f"创建知识图谱时出错: {e}")
            print(f"创建知识图谱时出错: {e}")
            return False

    def extract_relationships_from_knowledge(self, knowledge_points):
        """
        从知识点中提取关系

        参数:
            knowledge_points: 知识点列表

        返回:
            关系列表
        """
        # 首先提示模型生成一些通用关系
        relationships = self._generate_relationships_with_model(knowledge_points)

        # 然后从定义中提取隐含关系
        definition_relationships = self._extract_relationships_from_definitions(knowledge_points)

        # 合并关系列表
        relationships.extend(definition_relationships)

        print(f"从知识点中提取了 {len(relationships)} 个关系")
        return relationships

    def _generate_relationships_with_model(self, knowledge_points, max_concepts=30):
        """使用模型生成概念间的关系"""
        relationships = []

        # 如果知识点太多，只使用前max_concepts个
        concepts = [kp["concept"] for kp in knowledge_points[:max_concepts]]
        if not concepts:
            return []

        concepts_str = ", ".join([f'"{c}"' for c in concepts])

        prompt = f"""
分析以下概念之间的关系，并返回JSON格式的关系列表:
概念: {concepts_str}

请返回这些概念之间可能存在的关系，格式如下:
[
  {{
    "source": "源概念",
    "target": "目标概念",
    "relation": "关系类型",
    "strength": 0.8
  }}
]

关系类型包括:
- INCLUDES: 包含关系
- IS_PART_OF: 是...的一部分
- IS_PREREQUISITE_OF: 是...的前提
- IS_RELATED_TO: 与...相关
- REFERS_TO: 引用了
- SIMILAR_TO: 与...相似
"""

        response = self._generate_text(prompt)

        # 提取JSON
        try:
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                rels = json.loads(json_str)
                relationships.extend(rels)
            else:
                print("无法从关系生成响应中提取JSON")
        except Exception as e:
            print(f"解析关系JSON时出错: {e}")

        return relationships

    def _extract_relationships_from_definitions(self, knowledge_points):
        """从定义中提取隐含关系"""
        relationships = []

        for kp1 in knowledge_points:
            concept1 = kp1["concept"]
            definition1 = kp1.get("definition", "")

            for kp2 in knowledge_points:
                concept2 = kp2["concept"]

                # 避免自我关系
                if concept1 == concept2:
                    continue

                # 检查概念2是否出现在概念1的定义中
                if concept2 in definition1 and len(concept2) > 2:
                    # 添加一个引用关系
                    relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": "REFERS_TO",
                        "strength": 0.7
                    })

                # 检查概念包含关系
                if len(concept2) > 3 and concept2 in concept1 and concept2 != concept1:
                    relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": "INCLUDES",
                        "strength": 0.8
                    })

        return relationships