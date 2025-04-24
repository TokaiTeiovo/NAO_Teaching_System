# knowledge_extraction/manual_knowledge_builder.py
import json
import os


class ManualKnowledgeBuilder:
    """
    手动创建编译原理知识图谱
    """

    def __init__(self, output_path):
        self.output_path = output_path
        self.knowledge_points = []
        self.relationships = []

    def build_basic_knowledge_graph(self):
        """
        构建基本的编译原理知识图谱
        """
        print("开始构建编译原理知识图谱...")

        # 添加基本知识点
        self._add_compiler_basics()
        self._add_lexical_analysis()
        self._add_syntax_analysis()
        self._add_semantic_analysis()
        self._add_intermediate_code_generation()
        self._add_code_optimization()
        self._add_code_generation()

        # 添加关系
        self._add_relationships()

        # 保存知识图谱
        self._save_knowledge_graph()

        return len(self.knowledge_points), len(self.relationships)

    def _add_compiler_basics(self):
        """添加编译器基础知识点"""
        self.knowledge_points.extend([
            {
                "concept": "编译程序",
                "definition": "将用某种程序设计语言（源语言）编写的程序翻译成另一种语言（目标语言）的程序。",
                "type": "definition",
                "chapter": "第1章 引论",
                "importance": 5,
                "difficulty": 2
            },
            {
                "concept": "源程序",
                "definition": "用源语言编写的程序，是编译程序的输入。",
                "type": "definition",
                "chapter": "第1章 引论",
                "importance": 4,
                "difficulty": 1
            },
            {
                "concept": "目标程序",
                "definition": "编译程序的输出结果，用目标语言表示的程序。",
                "type": "definition",
                "chapter": "第1章 引论",
                "importance": 4,
                "difficulty": 1
            },
            {
                "concept": "编译过程",
                "definition": "从源程序到目标程序的转换过程，通常包括词法分析、语法分析、语义分析、中间代码生成、代码优化和目标代码生成等阶段。",
                "type": "definition",
                "chapter": "第1章 引论",
                "importance": 5,
                "difficulty": 3
            },
            {
                "concept": "编译程序结构",
                "definition": "编译程序通常由词法分析器、语法分析器、语义分析器、中间代码生成器、代码优化器和目标代码生成器等部分组成。",
                "type": "definition",
                "chapter": "第1章 引论",
                "importance": 5,
                "difficulty": 3
            }
        ])
        print("添加了编译器基础知识点")

    def _add_lexical_analysis(self):
        """添加词法分析相关知识点"""
        self.knowledge_points.extend([
            {
                "concept": "词法分析",
                "definition": "编译的第一阶段，将源程序字符流转换成标记（Token）序列的过程。",
                "type": "definition",
                "chapter": "第3章 词法分析",
                "importance": 5,
                "difficulty": 3
            },
            {
                "concept": "标记（Token）",
                "definition": "具有独立意义的最小语法单位，如标识符、关键字、常数、运算符等。",
                "type": "definition",
                "chapter": "第3章 词法分析",
                "importance": 5,
                "difficulty": 2
            },
            {
                "concept": "模式",
                "definition": "对标记的一种形式化描述，用正则表达式表示。",
                "type": "definition",
                "chapter": "第3章 词法分析",
                "importance": 4,
                "difficulty": 3
            },
            {
                "concept": "正则表达式",
                "definition": "用于描述正则语言的表达式，在词法分析中用来描述标记的模式。",
                "type": "definition",
                "chapter": "第3章 词法分析",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "有限自动机",
                "definition": "一种识别器，用于识别正则表达式所描述的语言，分为确定的有限自动机(DFA)和非确定的有限自动机(NFA)。",
                "type": "definition",
                "chapter": "第3章 词法分析",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "确定的有限自动机(DFA)",
                "definition": "一种特殊的有限自动机，其中任何一个状态对于任何输入符号都最多有一个转换。",
                "type": "definition",
                "chapter": "第3章 词法分析",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "非确定的有限自动机(NFA)",
                "definition": "一种有限自动机，其中一个状态对于一个输入符号可能有多个转换，或者可以在不接受任何输入的情况下转换（ε-转换）。",
                "type": "definition",
                "chapter": "第3章 词法分析",
                "importance": 5,
                "difficulty": 4
            }
        ])
        print("添加了词法分析相关知识点")

    def _add_syntax_analysis(self):
        """添加语法分析相关知识点"""
        self.knowledge_points.extend([
            {
                "concept": "语法分析",
                "definition": "编译的第二阶段，将词法分析得到的标记序列按照语法规则组织成语法树的过程。",
                "type": "definition",
                "chapter": "第4章 语法分析",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "上下文无关文法",
                "definition": "一种形式化的语法描述方法，由终结符、非终结符、产生式和开始符号组成，用于描述程序设计语言的语法结构。",
                "type": "definition",
                "chapter": "第2章 文法和语言",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "产生式",
                "definition": "上下文无关文法中的重写规则，通常表示为 A → α，其中 A 是一个非终结符，α 是终结符和非终结符的串。",
                "type": "definition",
                "chapter": "第2章 文法和语言",
                "importance": 5,
                "difficulty": 3
            },
            {
                "concept": "推导",
                "definition": "在上下文无关文法中，通过应用产生式规则将一个符号串转换为另一个符号串的过程。",
                "type": "definition",
                "chapter": "第2章 文法和语言",
                "importance": 4,
                "difficulty": 3
            },
            {
                "concept": "语法树",
                "definition": "表示推导过程的树状结构，内部节点表示非终结符，叶子节点表示终结符，根节点为文法的开始符号。",
                "type": "definition",
                "chapter": "第4章 语法分析",
                "importance": 5,
                "difficulty": 3
            },
            {
                "concept": "自顶向下分析",
                "definition": "从语法树的根节点（开始符号）开始，向下推导直到匹配输入串的语法分析方法。",
                "type": "definition",
                "chapter": "第4章 语法分析",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "自底向上分析",
                "definition": "从输入串开始，逐步归约到语法树的根节点（开始符号）的语法分析方法。",
                "type": "definition",
                "chapter": "第5章 自底向上分析技术",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "递归下降分析",
                "definition": "自顶向下分析的一种实现方法，为每个非终结符构造一个分析函数，函数间的相互调用反映了语法规则间的递归关系。",
                "type": "definition",
                "chapter": "第4章 语法分析",
                "importance": 4,
                "difficulty": 4
            },
            {
                "concept": "LL(1)分析法",
                "definition": "一种自顶向下的分析方法，能够在向前看一个输入符号的情况下确定使用哪个产生式进行推导。",
                "type": "definition",
                "chapter": "第4章 语法分析",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "LR分析法",
                "definition": "一种自底向上的分析方法，在向前看一个或多个输入符号的情况下进行规约，包括SLR、LR(1)和LALR(1)等变体。",
                "type": "definition",
                "chapter": "第5章 自底向上分析技术",
                "importance": 5,
                "difficulty": 5
            }
        ])
        print("添加了语法分析相关知识点")

    def _add_semantic_analysis(self):
        """添加语义分析相关知识点"""
        self.knowledge_points.extend([
            {
                "concept": "语义分析",
                "definition": "编译的第三阶段，检查源程序是否符合语言的语义规则，收集类型信息，并进行类型检查。",
                "type": "definition",
                "chapter": "第6章 语义分析",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "属性文法",
                "definition": "上下文无关文法的扩展，为文法符号增加了属性，并定义了计算属性值的规则，用于描述语言的静态语义。",
                "type": "definition",
                "chapter": "第6章 语义分析",
                "importance": 5,
                "difficulty": 5
            },
            {
                "concept": "综合属性",
                "definition": "一种属性，其值由其子节点的属性值确定，通常在自底向上的分析中计算。",
                "type": "definition",
                "chapter": "第6章 语义分析",
                "importance": 4,
                "difficulty": 4
            },
            {
                "concept": "继承属性",
                "definition": "一种属性，其值由其父节点和（或）兄弟节点的属性值确定，通常在自顶向下的分析中计算。",
                "type": "definition",
                "chapter": "第6章 语义分析",
                "importance": 4,
                "difficulty": 4
            },
            {
                "concept": "语法制导翻译",
                "definition": "一种将语义规则附加到语法规则上的翻译技术，通过遍历语法分析树并计算属性来实现语义处理。",
                "type": "definition",
                "chapter": "第6章 语义分析",
                "importance": 5,
                "difficulty": 5
            },
            {
                "concept": "符号表",
                "definition": "存储程序中标识符及其属性信息的数据结构，用于语义分析和代码生成阶段。",
                "type": "definition",
                "chapter": "第6章 语义分析",
                "importance": 5,
                "difficulty": 3
            }
        ])
        print("添加了语义分析相关知识点")

    def _add_intermediate_code_generation(self):
        """添加中间代码生成相关知识点"""
        self.knowledge_points.extend([
            {
                "concept": "中间代码",
                "definition": "介于源语言和目标语言之间的一种表示形式，便于后续的优化和代码生成。",
                "type": "definition",
                "chapter": "第7章 中间代码生成",
                "importance": 5,
                "difficulty": 3
            },
            {
                "concept": "三地址码",
                "definition": "一种中间代码形式，每条指令最多包含三个地址（两个操作数和一个结果）。",
                "type": "definition",
                "chapter": "第7章 中间代码生成",
                "importance": 5,
                "difficulty": 3
            },
            {
                "concept": "四元式",
                "definition": "三地址码的一种表示方法，每条指令由四部分组成：操作符、第一操作数、第二操作数和结果。",
                "type": "definition",
                "chapter": "第7章 中间代码生成",
                "importance": 4,
                "difficulty": 3
            },
            {
                "concept": "三元式",
                "definition": "三地址码的另一种表示方法，每条指令由三部分组成：操作符、第一操作数和第二操作数，结果隐含为该三元式本身。",
                "type": "definition",
                "chapter": "第7章 中间代码生成",
                "importance": 3,
                "difficulty": 3
            },
            {
                "concept": "语法树",
                "definition": "表示表达式语法结构的树状结构，是中间代码生成的常见中间形式。",
                "type": "definition",
                "chapter": "第7章 中间代码生成",
                "importance": 4,
                "difficulty": 3
            }
        ])
        print("添加了中间代码生成相关知识点")

    def _add_code_optimization(self):
        """添加代码优化相关知识点"""
        self.knowledge_points.extend([
            {
                "concept": "代码优化",
                "definition": "通过各种变换技术改进中间代码或目标代码，使之执行更快、占用空间更小或能耗更低。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 5,
                "difficulty": 5
            },
            {
                "concept": "局部优化",
                "definition": "在基本块内进行的优化，如常量折叠、代数简化、公共子表达式消除等。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 4,
                "difficulty": 4
            },
            {
                "concept": "全局优化",
                "definition": "跨越基本块的优化，如循环优化、全局公共子表达式消除、全局数据流分析等。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 5,
                "difficulty": 5
            },
            {
                "concept": "基本块",
                "definition": "程序中的一段顺序执行的指令序列，只有一个入口和一个出口。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 5,
                "difficulty": 3
            },
            {
                "concept": "控制流图",
                "definition": "表示程序控制流的有向图，节点为基本块，边表示可能的控制转移。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "数据流分析",
                "definition": "一种静态分析技术，研究程序中数据值如何沿着控制流传播，用于代码优化。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 5,
                "difficulty": 5
            },
            {
                "concept": "到达定值分析",
                "definition": "确定程序某点可能到达的变量定值的数据流分析。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 4,
                "difficulty": 5
            },
            {
                "concept": "活跃变量分析",
                "definition": "确定程序某点后可能被使用的变量的数据流分析。",
                "type": "definition",
                "chapter": "第8章 代码优化",
                "importance": 4,
                "difficulty": 5
            }
        ])
        print("添加了代码优化相关知识点")

    def _add_code_generation(self):
        """添加目标代码生成相关知识点"""
        self.knowledge_points.extend([
            {
                "concept": "目标代码生成",
                "definition": "将中间代码转换为目标机器的汇编代码或机器代码的过程。",
                "type": "definition",
                "chapter": "第9章 目标代码生成",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "寄存器分配",
                "definition": "决定哪些变量分配到寄存器，以及何时将寄存器内容写回内存的过程。",
                "type": "definition",
                "chapter": "第9章 目标代码生成",
                "importance": 5,
                "difficulty": 5
            },
            {
                "concept": "指令选择",
                "definition": "为中间代码的操作选择合适的目标机器指令序列的过程。",
                "type": "definition",
                "chapter": "第9章 目标代码生成",
                "importance": 5,
                "difficulty": 4
            },
            {
                "concept": "模式匹配",
                "definition": "一种指令选择技术，通过匹配中间代码中的模式来选择目标机器指令。",
                "type": "definition",
                "chapter": "第9章 目标代码生成",
                "importance": 4,
                "difficulty": 4
            },
            {
                "concept": "代码生成算法",
                "definition": "根据中间代码或抽象语法树生成目标代码的算法，考虑指令选择、寄存器分配和指令调度等因素。",
                "type": "definition",
                "chapter": "第9章 目标代码生成",
                "importance": 5,
                "difficulty": 5
            }
        ])
        print("添加了目标代码生成相关知识点")

    def _add_relationships(self):
        """添加知识点之间的关系"""
        self.relationships = [
            # 编译过程的顺序关系
            {"source": "编译过程", "target": "词法分析", "relation": "INCLUDES", "strength": 1.0},
            {"source": "编译过程", "target": "语法分析", "relation": "INCLUDES", "strength": 1.0},
            {"source": "编译过程", "target": "语义分析", "relation": "INCLUDES", "strength": 1.0},
            {"source": "编译过程", "target": "中间代码", "relation": "INCLUDES", "strength": 1.0},
            {"source": "编译过程", "target": "代码优化", "relation": "INCLUDES", "strength": 1.0},
            {"source": "编译过程", "target": "目标代码生成", "relation": "INCLUDES", "strength": 1.0},

            # 词法分析相关关系
            {"source": "词法分析", "target": "标记（Token）", "relation": "PRODUCES", "strength": 1.0},
            {"source": "词法分析", "target": "有限自动机", "relation": "USES", "strength": 0.9},
            {"source": "词法分析", "target": "正则表达式", "relation": "USES", "strength": 0.9},
            {"source": "正则表达式", "target": "模式", "relation": "DESCRIBES", "strength": 0.9},
            {"source": "有限自动机", "target": "确定的有限自动机(DFA)", "relation": "INCLUDES", "strength": 1.0},
            {"source": "有限自动机", "target": "非确定的有限自动机(NFA)", "relation": "INCLUDES", "strength": 1.0},

            # 语法分析相关关系
            {"source": "语法分析", "target": "上下文无关文法", "relation": "USES", "strength": 1.0},
            {"source": "语法分析", "target": "语法树", "relation": "PRODUCES", "strength": 1.0},
            {"source": "语法分析", "target": "自顶向下分析", "relation": "INCLUDES", "strength": 1.0},
            {"source": "语法分析", "target": "自底向上分析", "relation": "INCLUDES", "strength": 1.0},
            {"source": "自顶向下分析", "target": "递归下降分析", "relation": "INCLUDES", "strength": 1.0},
            {"source": "自顶向下分析", "target": "LL(1)分析法", "relation": "INCLUDES", "strength": 1.0},
            {"source": "自底向上分析", "target": "LR分析法", "relation": "INCLUDES", "strength": 1.0},
            {"source": "上下文无关文法", "target": "产生式", "relation": "CONSISTS_OF", "strength": 1.0},
            {"source": "上下文无关文法", "target": "推导", "relation": "SUPPORTS", "strength": 0.9},

            # 语义分析相关关系
            {"source": "语义分析", "target": "属性文法", "relation": "USES", "strength": 0.9},
            {"source": "语义分析", "target": "符号表", "relation": "USES", "strength": 1.0},
            {"source": "语义分析", "target": "语法制导翻译", "relation": "USES", "strength": 0.9},
            {"source": "属性文法", "target": "综合属性", "relation": "INCLUDES", "strength": 1.0},
            {"source": "属性文法", "target": "继承属性", "relation": "INCLUDES", "strength": 1.0},

            # 中间代码生成相关关系
            {"source": "中间代码", "target": "三地址码", "relation": "INCLUDES", "strength": 1.0},
            {"source": "三地址码", "target": "四元式", "relation": "REPRESENTED_BY", "strength": 0.8},
            {"source": "三地址码", "target": "三元式", "relation": "REPRESENTED_BY", "strength": 0.8},

            # 代码优化相关关系
            {"source": "代码优化", "target": "局部优化", "relation": "INCLUDES", "strength": 1.0},
            {"source": "代码优化", "target": "全局优化", "relation": "INCLUDES", "strength": 1.0},
            {"source": "代码优化", "target": "数据流分析", "relation": "USES", "strength": 0.9},
            {"source": "局部优化", "target": "基本块", "relation": "OPERATES_ON", "strength": 1.0},
            {"source": "全局优化", "target": "控制流图", "relation": "USES", "strength": 0.9},
            {"source": "数据流分析", "target": "到达定值分析", "relation": "INCLUDES", "strength": 1.0},
            {"source": "数据流分析", "target": "活跃变量分析", "relation": "INCLUDES", "strength": 1.0},

            # 目标代码生成相关关系
            {"source": "目标代码生成", "target": "寄存器分配", "relation": "INVOLVES", "strength": 1.0},
            {"source": "目标代码生成", "target": "指令选择", "relation": "INVOLVES", "strength": 1.0},
            {"source": "指令选择", "target": "模式匹配", "relation": "USES", "strength": 0.8},
            {"source": "目标代码生成", "target": "代码生成算法", "relation": "USES", "strength": 1.0},

            # 编译阶段的依赖关系
            {"source": "词法分析", "target": "语法分析", "relation": "IS_PREREQUISITE_OF", "strength": 1.0},
            {"source": "语法分析", "target": "语义分析", "relation": "IS_PREREQUISITE_OF", "strength": 1.0},
            {"source": "语义分析", "target": "中间代码", "relation": "IS_PREREQUISITE_OF", "strength": 1.0},
            {"source": "中间代码", "target": "代码优化", "relation": "IS_PREREQUISITE_OF", "strength": 0.9},
            {"source": "代码优化", "target": "目标代码生成", "relation": "IS_PREREQUISITE_OF", "strength": 0.9}
        ]
        print(f"添加了 {len(self.relationships)} 个关系")

    def _save_knowledge_graph(self):
        """
        保存知识图谱到JSON文件
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # 构建图谱数据
        graph_data = {
            "nodes": [],
            "links": []
        }

        # 添加节点
        for kp in self.knowledge_points:
            node_data = {
                "id": kp["concept"],
                "name": kp["concept"],
                "definition": kp.get("definition", ""),
                "type": kp.get("type", "Concept"),
                "chapter": kp.get("chapter", ""),
                "importance": kp.get("importance", 3),
                "difficulty": kp.get("difficulty", 3)
            }
            graph_data["nodes"].append(node_data)

        # 添加关系
        for rel in self.relationships:
            link_data = {
                "source": rel["source"],
                "target": rel["target"],
                "type": rel["relation"],
                "strength": rel.get("strength", 0.5)
            }
            graph_data["links"].append(link_data)

        # 保存到文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        print(f"知识图谱已保存到: {self.output_path}")
        print(f"节点数量: {len(graph_data['nodes'])}")
        print(f"关系数量: {len(graph_data['links'])}")