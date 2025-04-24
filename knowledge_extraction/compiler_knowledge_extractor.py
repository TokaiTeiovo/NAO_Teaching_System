# knowledge_extraction/compiler_knowledge_extractor.py
import re
import logging
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('compiler_knowledge_extractor')

# 确保NLTK数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下载NLTK数据...")
    nltk.download('punkt')


class CompilerKnowledgeExtractor:
    """
    编译原理知识点提取器
    """

    def __init__(self):
        # 定义编译原理领域的特定模式
        self.definition_patterns = [
            r'([^。；,.!?]+)是指([^。；,.!?]+)[。；,.!?]',  # X是指Y
            r'([^。；,.!?]+)被定义为([^。；,.!?]+)[。；,.!?]',  # X被定义为Y
            r'([^。；,.!?]+)定义为([^。；,.!?]+)[。；,.!?]',  # X定义为Y
            r'([^。；,.!?]+)(指|表示)([^。；,.!?]+)[。；,.!?]',  # X指/表示Y
            r'所谓([^，,]+)[，,]([^。；,.!?]+)[。；,.!?]',  # 所谓X，Y
            r'([^。；,.!?]+)的定义是([^。；,.!?]+)[。；,.!?]',  # X的定义是Y
            r'定义([^。；,.!?]+)为([^。；,.!?]+)[。；,.!?]',  # 定义X为Y
            r'([^。；,.!?]+)(称为|叫做)([^。；,.!?]+)[。；,.!?]'  # X称为/叫做Y
        ]

        # 编译原理关键术语
        self.compiler_terms = [
            "编译", "编译程序", "词法分析", "语法分析", "语义分析",
            "代码生成", "中间代码", "目标代码", "优化", "语法树",
            "符号表", "标记", "词素", "正则表达式", "有限自动机",
            "DFA", "NFA", "上下文无关文法", "产生式", "推导",
            "递归下降", "预测分析", "LL分析", "LR分析", "语法制导",
            "属性文法", "语义规则", "三地址码", "四元式", "控制流图",
            "基本块", "数据流分析", "寄存器分配", "死代码"
        ]

    def extract_definitions(self, text):
        """
        提取定义型知识点
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
        for sentence in tqdm(sentences, desc="提取定义", unit="句"):
            # 对每个模式进行匹配
            for pattern in self.definition_patterns:
                matches = re.finditer(pattern, sentence)

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

                    # 检查概念是否有效
                    if len(concept) < 2 or len(definition) < 5:
                        continue

                    # 保存定义
                    definitions.append({
                        "concept": concept,
                        "definition": definition,
                        "source_text": sentence,
                        "type": "definition"
                    })

        logger.info(f"提取了 {len(definitions)} 个定义")
        print(f"提取了 {len(definitions)} 个定义")
        return definitions

    def extract_key_terms(self, text):
        """
        提取关键术语
        """
        if not text:
            logger.warning("文本为空，无法提取关键术语")
            return []

        # 分割文本为句子
        sentences = sent_tokenize(text)

        # 在文本中查找编译原理术语
        print("开始提取关键术语...")
        terms = []
        for term in tqdm(self.compiler_terms, desc="提取关键术语", unit="术语"):
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, text)

            count = 0
            for _ in matches:
                count += 1

            if count > 0:
                # 计算权重 - 出现频率
                weight = min(1.0, count / 20.0)  # 最多出现20次算作权重1.0

                terms.append({
                    "term": term,
                    "count": count,
                    "weight": weight,
                    "type": "term"
                })

        # 按出现次数排序
        terms.sort(key=lambda x: x["count"], reverse=True)

        logger.info(f"提取了 {len(terms)} 个关键术语")
        print(f"提取了 {len(terms)} 个关键术语")
        return terms

    def extract_compiler_knowledge(self, text, chapter=""):
        """
        提取编译原理知识点
        """
        print(f"\n从章节 '{chapter}' 提取知识点...")

        # 提取定义
        definitions = self.extract_definitions(text)

        # 提取关键术语
        terms = self.extract_key_terms(text)

        # 合并知识点
        knowledge_points = []

        # 添加定义型知识点
        for definition in definitions:
            # 添加章节信息
            definition["chapter"] = chapter
            knowledge_points.append(definition)

        # 添加术语型知识点
        for term in terms:
            # 检查是否已经作为定义添加
            if not any(kp["concept"] == term["term"] for kp in knowledge_points):
                knowledge_points.append({
                    "concept": term["term"],
                    "definition": "",  # 暂无定义
                    "type": "term",
                    "weight": term["weight"],
                    "count": term["count"],
                    "chapter": chapter
                })

        logger.info(f"从章节 '{chapter}' 总共提取了 {len(knowledge_points)} 个知识点")
        print(f"从章节 '{chapter}' 总共提取了 {len(knowledge_points)} 个知识点")
        return knowledge_points