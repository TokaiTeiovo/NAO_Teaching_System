# knowledge_extraction/knowledge_extractor.py
import re
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('knowledge_extractor')

# 确保NLTK数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下载NLTK数据...")
    nltk.download('punkt')


class KnowledgeExtractor:
    """
    从文本中提取知识点的工具类
    """

    def __init__(self):
        """
        初始化知识点提取器
        """
        # 定义关键词模式
        self.keyword_patterns = [
            r'([^。；,.!?;]+)是指([^。；,.!?;]+)[。；,.!?;]',  # X是指Y
            r'([^。；,.!?;]+)称为([^。；,.!?;]+)[。；,.!?;]',  # X称为Y
            r'([^。；,.!?;]+)定义为([^。；,.!?;]+)[。；,.!?;]',  # X定义为Y
            r'([^。；,.!?;]+)叫做([^。；,.!?;]+)[。；,.!?;]',  # X叫做Y
            r'所谓([^，,]+)[，,]?([^。；,.!?;]+)[。；,.!?;]',  # 所谓X，Y
            r'([^。；,.!?;]+)(是|指|表示)([^。；,.!?;]+)[。；,.!?;]',  # X是/指/表示Y
            r'定义[^\n]*[:：]\s*([^。；,.!?;]+)[。；,.!?;]',  # 定义X:
            r'([^。；,.!?;，,]+)的定义是\s*[:：]?\s*([^。；,.!?;]+)[。；,.!?;]',  # X的定义是
            r'([^。；,.!?;，,]+)被定义为\s*[:：]?\s*([^。；,.!?;]+)[。；,.!?;]',  # X被定义为
            r'概念\s*\d*\s*[:：]\s*([^。；,.!?;]+)[。；,.!?;]',  # 概念X:
            r'([^。；,.!?;，,]+)指的是\s*([^。；,.!?;]+)[。；,.!?;]'  # X指的是
        ]

        # 编译原理领域关键词
        self.compiler_keywords = [
            "编译程序", "源程序", "目标程序", "词法分析", "语法分析", "语义分析",
            "中间代码", "代码优化", "目标代码生成", "语法树", "符号表", "标记", "词素",
            "正则表达式", "有限自动机", "DFA", "NFA", "上下文无关文法", "产生式",
            "递归下降", "LL分析", "LR分析", "LALR", "语义规则", "属性文法", "三地址码",
            "四元式", "控制流图", "基本块", "数据流分析", "寄存器分配", "代码生成"
        ]

    def extract_definitions(self, text):
        """
        从文本中提取定义型知识点
        """
        if not text:
            logger.warning("文本为空，无法提取定义")
            return []

        definitions = []

        # 分割文本为句子
        print("分割文本为句子...")
        sentences = sent_tokenize(text)
        logger.info(f"文本分割为 {len(sentences)} 个句子")
        print(f"文本分割为 {len(sentences)} 个句子")

        # 遍历每个句子
        print("开始提取定义...")
        for sentence in tqdm(sentences, desc="提取定义"):
            for pattern in self.keyword_patterns:
                matches = list(re.finditer(pattern, sentence))
                for match in matches:
                    groups = match.groups()

                    # 根据模式类型处理匹配结果
                    if len(groups) == 1:
                        # 模式中只有一个捕获组，如"定义: X"
                        concept_definition = groups[0].strip()
                        # 尝试从定义中分离概念和解释
                        parts = re.split(r'[是指表示]', concept_definition, 1)
                        if len(parts) > 1:
                            concept = parts[0].strip()
                            definition = parts[1].strip()
                        else:
                            concept = "未知概念"
                            definition = concept_definition
                    elif len(groups) == 2:
                        # 模式中有两个捕获组，如"X是指Y"
                        concept = groups[0].strip()
                        definition = groups[1].strip()
                    elif len(groups) == 3:
                        # 模式中有三个捕获组，如"X是Y"
                        concept = groups[0].strip()
                        definition = groups[2].strip()
                    else:
                        continue

                    # 过滤掉过短的概念或定义
                    if len(concept) > 1 and len(definition) > 5:
                        # 检查是否与编译原理相关
                        is_compiler_related = False
                        for keyword in self.compiler_keywords:
                            if keyword in concept or keyword in definition:
                                is_compiler_related = True
                                break

                        if is_compiler_related or len(self.compiler_keywords) == 0:
                            definitions.append({
                                "concept": concept,
                                "definition": definition,
                                "source_text": match.group(0),
                                "type": "definition"
                            })

        logger.info(f"提取了 {len(definitions)} 个定义")
        print(f"提取了 {len(definitions)} 个定义")
        return definitions

    def extract_key_terms(self, text, min_freq=2, max_terms=50):
        """
        从文本中提取关键术语
        """
        if not text:
            logger.warning("文本为空，无法提取关键术语")
            return []

        # 使用关键词匹配提取术语
        print("提取关键术语...")
        key_terms = []
        for term in tqdm(self.compiler_keywords, desc="匹配关键术语"):
            # 使用正则表达式匹配整个单词
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = list(re.finditer(pattern, text))
            count = len(matches)

            if count >= min_freq:
                # 计算词频得分
                score = min(1.0, count / 50.0)  # 最多出现50次算满分

                key_terms.append({
                    "term": term,
                    "score": float(score),
                    "count": count,
                    "type": "key_term"
                })

        # 按得分排序
        key_terms.sort(key=lambda x: x["score"], reverse=True)

        # 限制数量
        key_terms = key_terms[:max_terms]

        logger.info(f"提取了 {len(key_terms)} 个关键术语")
        print(f"提取了 {len(key_terms)} 个关键术语")
        return key_terms

    def extract_relationships(self, definitions):
        """
        从定义中提取可能的关系
        """
        print("推断知识点之间的关系...")
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
            (r'是.*的一部分', 'IS_PART_OF')
        ]

        # 从定义中推断关系
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
                if len(concept2) > 3 and concept2 in concept1:
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
        """
        if not text:
            logger.warning("文本为空，无法提取知识点")
            return []

        logger.info(f"从章节 '{chapter_title}' 提取知识点...")
        print(f"从章节 '{chapter_title}' 提取知识点...")

        # 提取定义
        definitions = self.extract_definitions(text)
        # 提取关键术语
        key_terms = self.extract_key_terms(text)

        # 合并知识点
        knowledge_points = []

        # 添加定义型知识点
        for definition in definitions:
            knowledge_points.append({
                "concept": definition["concept"],
                "definition": definition["definition"],
                "type": "definition",
                "chapter": chapter_title,
                "source_text": definition["source_text"]
            })

        # 添加术语型知识点
        for term in key_terms:
            # 检查是否已经作为定义添加
            if not any(kp["concept"] == term["term"] for kp in knowledge_points):
                knowledge_points.append({
                    "concept": term["term"],
                    "definition": "",  # 暂无定义
                    "type": "term",
                    "score": term["score"],
                    "chapter": chapter_title
                })

        return knowledge_points