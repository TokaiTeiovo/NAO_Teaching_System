# knowledge_extraction/knowledge_extractor.py
import re
import logging
from tqdm import tqdm

# 创建日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('knowledge_extractor')


class KnowledgeExtractor:
    """从文本中提取知识点的工具类"""

    def __init__(self, domain_keywords=None):
        """
        初始化知识点提取器

        参数:
            domain_keywords: 领域关键词列表，用于过滤相关知识点
        """
        # 定义用于识别定义句的模式
        self.definition_patterns = [
            r'([^。；,.!?]+)是指([^。；,.!?]+)[。；,.!?]',  # X是指Y
            r'([^。；,.!?]+)称为([^。；,.!?]+)[。；,.!?]',  # X称为Y
            r'([^。；,.!?]+)定义为([^。；,.!?]+)[。；,.!?]',  # X定义为Y
            r'([^。；,.!?]+)被称为([^。；,.!?]+)[。；,.!?]',  # X被称为Y
            r'所谓([^，,]+)[，,]?([^。；,.!?;]+)[。；,.!?;]',  # 所谓X，Y
            r'([^。；,.!?;]+)(是|指|表示)([^。；,.!?;]+)[。；,.!?;]',  # X是/指/表示Y
            r'定义\s*\d*\s*[:：]\s*([^。；,.!?;]+)[。；,.!?;]',  # 定义X:
            r'([^。；,.!?;，,]+)的定义是\s*[:：]?\s*([^。；,.!?;]+)[。；,.!?;]',  # X的定义是
        ]

        # 使用提供的领域关键词或默认为空列表
        self.domain_keywords = domain_keywords or []

    def simple_sentence_tokenize(self, text):
        """
        简单的句子分割函数，替代NLTK的sent_tokenize

        参数:
            text: 输入文本

        返回:
            句子列表
        """
        # 使用常见的句子结束符号分割文本
        sentences = re.split(r'[。！？.!?;；]+', text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def extract_definitions(self, text):
        """
        从文本中提取定义型知识点

        参数:
            text: 输入文本

        返回:
            定义列表，每个定义是一个字典，包含概念和定义
        """
        if not text:
            logger.warning("文本为空，无法提取定义")
            return []

        definitions = []

        # 分割文本为句子
        print("分割文本为句子...")
        sentences = self.simple_sentence_tokenize(text)
        logger.info(f"文本分割为 {len(sentences)} 个句子")
        print(f"文本分割为 {len(sentences)} 个句子")

        # 遍历每个句子
        print("开始提取定义...")
        for sentence in tqdm(sentences, desc="提取定义"):
            # 为每个句子添加结束符，以便模式匹配
            sentence_with_period = sentence + "。"
            for pattern in self.definition_patterns:
                matches = re.finditer(pattern, sentence_with_period)
                for match in matches:
                    groups = match.groups()

                    # 根据模式类型处理匹配结果
                    if len(groups) == 2:
                        concept = groups[0].strip()
                        definition = groups[1].strip()
                    elif len(groups) == 3:
                        concept = groups[0].strip()
                        definition = groups[2].strip()
                    else:
                        continue

                    # 过滤掉过短的概念或定义
                    if len(concept) < 2 or len(definition) < 5:
                        continue

                    # 如果有领域关键词，检查是否相关
                    is_relevant = True
                    if self.domain_keywords:
                        is_relevant = False
                        for keyword in self.domain_keywords:
                            if keyword in concept or keyword in definition:
                                is_relevant = True
                                break

                    if is_relevant:
                        definitions.append({
                            "concept": concept,
                            "definition": definition,
                            "source_text": match.group(0),
                            "type": "definition"
                        })

        logger.info(f"提取了 {len(definitions)} 个定义")
        print(f"提取了 {len(definitions)} 个定义")
        return definitions

    def extract_relationships(self, definitions):
        """
        从定义中提取可能的关系

        参数:
            definitions: 定义列表

        返回:
            关系列表，每个关系是一个字典，包含源概念、目标概念和关系类型
        """
        relationships = []

        # 定义常见关系类型
        relation_patterns = [
            (r'包括', 'INCLUDES'),
            (r'包含', 'INCLUDES'),
            (r'属于', 'BELONGS_TO'),
            (r'组成', 'CONSISTS_OF'),
            (r'使用', 'USES'),
            (r'产生', 'PRODUCES'),
            (r'生成', 'GENERATES'),
            (r'是.*一种', 'IS_A'),
            (r'是.*的一部分', 'IS_PART_OF'),
            (r'基于', 'BASED_ON'),
            (r'依赖', 'DEPENDS_ON'),
            (r'需要', 'REQUIRES'),
            (r'转换', 'TRANSFORMS_TO')
        ]

        # 从定义中推断关系
        print("推断知识点之间的关系...")
        for def1 in tqdm(definitions, desc="分析关系"):
            concept1 = def1["concept"]
            definition1 = def1["definition"]

            # 检查定义中是否包含其他概念
            for def2 in definitions:
                concept2 = def2["concept"]

                # 跳过自身
                if concept1 == concept2:
                    continue

                # 如果概念2在概念1的定义中
                if concept2 in definition1:
                    # 尝试识别关系类型
                    relation_type = "RELATED_TO"  # 默认关系
                    relation_strength = 0.5  # 默认强度

                    # 通过模式判断关系类型
                    for pattern, rel_type in relation_patterns:
                        if re.search(pattern, definition1):
                            relation_type = rel_type
                            relation_strength = 0.7
                            break

                    # 添加关系
                    relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": relation_type,
                        "strength": relation_strength
                    })

                # 检查概念包含关系
                if len(concept2) > 3 and concept2 in concept1 and concept2 != concept1:
                    relationships.append({
                        "source": concept1,
                        "target": concept2,
                        "relation": "INCLUDES",
                        "strength": 0.6
                    })

        logger.info(f"推断了 {len(relationships)} 个关系")
        print(f"推断了 {len(relationships)} 个关系")
        return relationships

    def extract_knowledge_points(self, text, chapter_title=""):
        """
        从文本中提取所有知识点

        参数:
            text: 输入文本
            chapter_title: 章节标题

        返回:
            知识点列表
        """
        if not text:
            logger.warning("文本为空，无法提取知识点")
            return []

        logger.info(f"从章节 '{chapter_title}' 提取知识点...")
        print(f"从章节 '{chapter_title}' 提取知识点...")

        # 提取定义
        definitions = self.extract_definitions(text)

        # 为每个定义添加章节信息
        for definition in definitions:
            definition["chapter"] = chapter_title

        logger.info(f"从章节 '{chapter_title}' 总共提取了 {len(definitions)} 个知识点")
        print(f"从章节 '{chapter_title}' 总共提取了 {len(definitions)} 个知识点")
        return definitions