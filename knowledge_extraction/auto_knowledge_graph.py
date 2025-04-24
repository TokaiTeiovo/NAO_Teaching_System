# knowledge_extraction/auto_knowledge_graph.py
import os
import json
import logging
import networkx as nx
from tqdm import tqdm
from knowledge_extraction.ocr_pdf_extractor import OCRPDFExtractor
from knowledge_extraction.compiler_knowledge_extractor import CompilerKnowledgeExtractor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('auto_knowledge_graph')


class AutoKnowledgeGraph:
    """
    自动知识图谱生成器
    """

    def __init__(self, pdf_path, output_path):
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.text_content = ""
        self.chapters = {}
        self.knowledge_points = []
        self.relationships = []
        self.graph = nx.DiGraph()  # 知识图谱

        logger.info(f"初始化自动知识图谱生成器: {pdf_path}")
        print(f"初始化自动知识图谱生成器: {pdf_path}")

    def extract_text_with_ocr(self, sample_pages=True):
        """
        使用OCR提取文本

        参数:
            sample_pages: 是否只处理部分页面作为样本
        """
        logger.info("开始使用OCR提取文本...")
        print("\n=== 步骤1：使用OCR提取文本 ===")

        # 初始化OCR提取器
        extractor = OCRPDFExtractor(self.pdf_path)

        # 如果只处理样本页面，则只提取前20页
        if sample_pages:
            logger.info("只处理前20页作为样本...")
            print("只处理前20页作为样本...")
            text = extractor.extract_text(0, 20)
        else:
            text = extractor.extract_text()

        self.text_content = text

        # 提取章节
        self.chapters = extractor.extract_chapters()

        return text

    def extract_knowledge_points(self):
        """
        提取知识点
        """
        logger.info("开始提取知识点...")
        print("\n=== 步骤2：提取知识点 ===")

        # 初始化知识点提取器
        knowledge_extractor = CompilerKnowledgeExtractor()

        # 如果有章节，则按章节提取
        if self.chapters:
            # 使用tqdm创建进度条
            for chapter_title, chapter_info in tqdm(self.chapters.items(), desc="处理章节", unit="章"):
                logger.info(f"从章节提取知识点: {chapter_title}")
                chapter_text = chapter_info["text"]

                # 提取知识点
                knowledge_points = knowledge_extractor.extract_compiler_knowledge(
                    chapter_text, chapter_title
                )

                self.knowledge_points.extend(knowledge_points)
        else:
            # 直接从全文提取
            logger.info("从全文提取知识点")
            print("从全文提取知识点")
            knowledge_points = knowledge_extractor.extract_compiler_knowledge(
                self.text_content, "全文"
            )

            self.knowledge_points.extend(knowledge_points)

        logger.info(f"总共提取了 {len(self.knowledge_points)} 个知识点")
        print(f"总共提取了 {len(self.knowledge_points)} 个知识点")
        return self.knowledge_points

    def infer_relationships(self):
        """
        推断知识点间的关系
        """
        logger.info("开始推断知识点间的关系...")
        print("\n=== 步骤3：推断知识点间的关系 ===")

        # 编译过程的标准阶段
        compiler_phases = [
            "词法分析", "语法分析", "语义分析", "中间代码生成", "代码优化", "目标代码生成"
        ]

        # 添加编译过程阶段的顺序关系
        for i in range(len(compiler_phases) - 1):
            self.relationships.append({
                "source": compiler_phases[i],
                "target": compiler_phases[i + 1],
                "relation": "IS_PREREQUISITE_OF",
                "strength": 1.0
            })

            # 添加与编译过程的关系
            self.relationships.append({
                "source": "编译过程",
                "target": compiler_phases[i],
                "relation": "INCLUDES",
                "strength": 1.0
            })

        # 添加最后一个阶段与编译过程的关系
        self.relationships.append({
            "source": "编译过程",
            "target": compiler_phases[-1],
            "relation": "INCLUDES",
            "strength": 1.0
        })

        # 基于定义推断关系
        print("推断知识点之间的关系...")
        # 只处理定义类型的知识点
        definition_points = [kp for kp in self.knowledge_points if kp["type"] == "definition"]

        # 使用tqdm创建进度条
        for kp1 in tqdm(definition_points, desc="分析定义关系", unit="概念"):
            concept1 = kp1["concept"]
            definition1 = kp1["definition"]

            for kp2 in self.knowledge_points:
                if kp1 == kp2:
                    continue

                concept2 = kp2["concept"]

                # 检查概念是否出现在另一个概念的定义中
                if concept2 in definition1:
                    self.relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": "REFERS_TO",
                        "strength": 0.8
                    })

                # 检查概念的包含关系
                if len(concept2) > 3 and concept2 in concept1:
                    self.relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": "INCLUDES",
                        "strength": 0.7
                    })

        # 预定义的关系
        print("添加预定义关系...")
        predefined_relations = [
            {"source": "编译程序", "target": "源程序", "relation": "PROCESSES", "strength": 1.0},
            {"source": "编译程序", "target": "目标程序", "relation": "PRODUCES", "strength": 1.0},
            {"source": "词法分析", "target": "标记", "relation": "PRODUCES", "strength": 1.0},
            {"source": "词法分析", "target": "正则表达式", "relation": "USES", "strength": 0.9},
            {"source": "词法分析", "target": "有限自动机", "relation": "USES", "strength": 0.9},
            {"source": "语法分析", "target": "语法树", "relation": "PRODUCES", "strength": 1.0},
            {"source": "语法分析", "target": "上下文无关文法", "relation": "USES", "strength": 1.0},
            {"source": "语义分析", "target": "符号表", "relation": "USES", "strength": 1.0},
            {"source": "有限自动机", "target": "DFA", "relation": "INCLUDES", "strength": 1.0},
            {"source": "有限自动机", "target": "NFA", "relation": "INCLUDES", "strength": 1.0}
        ]

        # 添加预定义关系
        self.relationships.extend(predefined_relations)

        logger.info(f"总共推断了 {len(self.relationships)} 个关系")
        print(f"总共推断了 {len(self.relationships)} 个关系")
        return self.relationships

    def build_knowledge_graph(self):
        """
        构建知识图谱
        """
        logger.info("开始构建知识图谱...")
        print("\n=== 步骤4：构建知识图谱 ===")

        # 添加节点
        print("添加节点到知识图谱...")
        for kp in tqdm(self.knowledge_points, desc="添加节点", unit="节点"):
            # 节点属性
            node_attrs = {
                "name": kp["concept"],
                "type": "Concept",
                "definition": kp.get("definition", ""),
                "chapter": kp.get("chapter", ""),
                "importance": 3,  # 默认重要性
                "difficulty": 3  # 默认难度
            }

            # 添加到图
            self.graph.add_node(kp["concept"], **node_attrs)

        # 添加关系
        print("添加关系到知识图谱...")
        for rel in tqdm(self.relationships, desc="添加关系", unit="关系"):
            source = rel["source"]
            target = rel["target"]

            # 确保源节点和目标节点存在
            if source not in self.graph:
                self.graph.add_node(source, name=source, type="Concept")

            if target not in self.graph:
                self.graph.add_node(target, name=target, type="Concept")

            # 添加关系
            self.graph.add_edge(
                source, target,
                type=rel["relation"],
                strength=rel.get("strength", 0.5)
            )

        logger.info(f"知识图谱构建完成，共 {len(self.graph.nodes)} 个节点和 {len(self.graph.edges)} 条边")
        print(f"知识图谱构建完成，共 {len(self.graph.nodes)} 个节点和 {len(self.graph.edges)} 条边")
        return self.graph

    def save_knowledge_graph(self):
        """
        保存知识图谱为JSON文件
        """
        logger.info(f"保存知识图谱到: {self.output_path}")
        print("\n=== 步骤5：保存知识图谱 ===")
        print(f"保存知识图谱到: {self.output_path}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # 构建图谱数据
        graph_data = {
            "nodes": [],
            "links": []
        }

        # 添加节点
        print("处理节点数据...")
        for node, attrs in tqdm(self.graph.nodes(data=True), desc="处理节点", unit="节点"):
            node_data = {
                "id": node,
                **attrs
            }
            graph_data["nodes"].append(node_data)

        # 添加关系
        print("处理关系数据...")
        for source, target, attrs in tqdm(self.graph.edges(data=True), desc="处理关系", unit="关系"):
            link_data = {
                "source": source,
                "target": target,
                **attrs
            }
            graph_data["links"].append(link_data)

        # 保存到文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        logger.info(f"知识图谱已保存，节点数量: {len(graph_data['nodes'])}, 关系数量: {len(graph_data['links'])}")
        print(f"知识图谱已成功保存，节点数量: {len(graph_data['nodes'])}, 关系数量: {len(graph_data['links'])}")
        return True

    def generate(self, use_ocr=True, sample_pages=True):
        """
        生成知识图谱

        参数:
            use_ocr: 是否使用OCR提取文本
            sample_pages: 是否只处理部分页面
        """
        print("\n===== 开始生成编译原理知识图谱 =====")

        # 1. 提取文本
        if use_ocr:
            self.extract_text_with_ocr(sample_pages)

        # 2. 提取知识点
        if self.text_content:
            self.extract_knowledge_points()
        else:
            # 如果没有文本，使用预定义的知识点
            logger.warning("没有提取到文本，使用预定义的知识点")
            print("\n警告：没有提取到文本，使用预定义的知识点")
            self._use_predefined_knowledge()

        # 3. 推断关系
        self.infer_relationships()

        # 4. 构建知识图谱
        self.build_knowledge_graph()

        # 5. 保存知识图谱
        self.save_knowledge_graph()

        print("\n===== 知识图谱生成完成 =====")
        return True

    def _use_predefined_knowledge(self):
        """
        使用预定义的编译原理知识点
        """
        print("加载预定义知识点...")
        predefined_knowledge = [
            {"concept": "编译程序", "definition": "将源程序翻译成目标程序的程序", "type": "definition"},
            {"concept": "源程序", "definition": "用源语言编写的程序", "type": "definition"},
            {"concept": "目标程序", "definition": "编译程序的输出结果", "type": "definition"},
            {"concept": "编译过程", "definition": "从源程序到目标程序的转换过程", "type": "definition"},
            {"concept": "词法分析", "definition": "将源程序分解成单词符号的过程", "type": "definition"},
            {"concept": "语法分析", "definition": "在词法分析的基础上识别句子结构的过程", "type": "definition"},
            {"concept": "语义分析", "definition": "检查程序是否有语义错误并收集类型信息的过程", "type": "definition"},
            {"concept": "中间代码生成", "definition": "将程序翻译成与机器无关的中间表示形式的过程",
             "type": "definition"},
            {"concept": "代码优化", "definition": "对中间代码进行变换，使生成的目标代码更高效", "type": "definition"},
            {"concept": "目标代码生成", "definition": "将中间代码映射到目标机器上的过程", "type": "definition"}
        ]

        self.knowledge_points = predefined_knowledge
        logger.info(f"使用了 {len(predefined_knowledge)} 个预定义知识点")
        print(f"使用了 {len(predefined_knowledge)} 个预定义知识点")