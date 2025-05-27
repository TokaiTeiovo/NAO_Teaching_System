#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - 知识图谱提取独立启动器
包含: PDF处理、OCR提取、LLM知识提取、Neo4j导入
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 导入知识提取模块
try:
    from knowledge_extractor.knowledge_extractor_integrated import main as extractor_main
except ImportError:
    print("❌ 无法导入知识提取模块，请检查路径配置")
    sys.exit(1)


def print_banner():
    """显示启动横幅"""
    banner = """
╔══════════════════════════════════════════════════╗
║        📚 知识图谱提取器 - 独立启动器             ║
║                                                  ║
║  • PDF文档OCR处理                                ║
║  • 图片文件夹批量处理                            ║
║  • LLM智能知识提取                               ║
║  • Neo4j图数据库导入                             ║
╚══════════════════════════════════════════════════╝
"""
    print(banner)


def check_requirements():
    """检查运行环境"""
    print("🔍 检查运行环境...")

    # 检查必要的目录
    required_dirs = [
        PROJECT_ROOT / "shared",
        PROJECT_ROOT / "shared" / "temp",
        PROJECT_ROOT / "shared" / "output",
        PROJECT_ROOT / "shared" / "models",
        PROJECT_ROOT / "logs"
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ 创建目录: {dir_path}")
        else:
            print(f"   ✅ 目录存在: {dir_path}")

    # 检查关键依赖
    missing_deps = []

    try:
        import torch
        print("   ✅ PyTorch 已安装")
    except ImportError:
        missing_deps.append("torch")

    try:
        import transformers
        print("   ✅ Transformers 已安装")
    except ImportError:
        missing_deps.append("transformers")

    try:
        from paddleocr import PaddleOCR
        print("   ✅ PaddleOCR 已安装")
    except ImportError:
        print("   ⚠️  PaddleOCR 未安装，将尝试使用EasyOCR")
        try:
            import easyocr
            print("   ✅ EasyOCR 已安装")
        except ImportError:
            missing_deps.append("paddleocr 或 easyocr")

    try:
        from pdf2image import convert_from_path
        print("   ✅ pdf2image 已安装")
    except ImportError:
        missing_deps.append("pdf2image")

    if missing_deps:
        print(f"\n❌ 缺少必要依赖:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\n请安装缺失的依赖包后再试")
        return False

    print("✅ 环境检查完成")
    return True


def validate_input_files(args):
    """验证输入文件"""
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"❌ PDF文件不存在: {args.pdf}")
            return False
        print(f"   ✅ PDF文件: {pdf_path}")

    if args.images:
        images_path = Path(args.images)
        if not images_path.exists() or not images_path.is_dir():
            print(f"❌ 图片文件夹不存在: {args.images}")
            return False
        print(f"   ✅ 图片文件夹: {images_path}")

    if args.json:
        json_path = Path(args.json)
        # 检查多个可能的位置
        possible_paths = [
            json_path,
            PROJECT_ROOT / args.json,
            PROJECT_ROOT / "shared" / args.json,
            PROJECT_ROOT / "shared" / "temp" / args.json,
            PROJECT_ROOT / "shared" / "output" / args.json
        ]

        found = False
        for path in possible_paths:
            if path.exists():
                print(f"   ✅ JSON文件: {path}")
                found = True
                break

        if not found:
            print(f"❌ JSON文件不存在: {args.json}")
            return False

    return True


def show_processing_plan(args):
    """显示处理计划"""
    print("\n📋 处理计划:")

    if args.pdf:
        print(f"   📄 输入源: PDF文件 ({args.pdf})")
    elif args.images:
        print(f"   🖼️  输入源: 图片文件夹 ({args.images})")
    elif args.json:
        print(f"   📝 输入源: JSON文件 ({args.json})")

    print(f"   🎯 知识领域: {args.domain}")
    print(f"   📦 批次大小: {args.batch_size}")

    if args.start_index > 0:
        print(f"   🚀 开始索引: {args.start_index}")

    if args.max_items:
        print(f"   🔢 最大处理数: {args.max_items}")

    print(f"   💾 输出文件: {args.output}")

    if args.pdf:
        print(f"   🖼️  图像DPI: {args.dpi}")
        print(f"   💾 保存图像: {'是' if args.save_images and not args.no_save_images else '否'}")

    print(f"   🔍 OCR引擎: {args.ocr_engine}")
    print(f"   🤖 使用GPU: {'是' if args.use_gpu else '否'}")

    if args.import_neo4j:
        print(f"   🗄️  导入Neo4j: 是 ({args.neo4j_uri})")
    else:
        print(f"   🗄️  导入Neo4j: 否")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多模态智能教学系统 - 知识图谱提取器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从PDF提取知识图谱
  python start_knowledge_extractor.py --pdf "document.pdf" --domain "计算机科学"

  # 从图片文件夹提取
  python start_knowledge_extractor.py --images "pic_folder" --batch-size 5

  # 从JSON文件生成知识图谱
  python start_knowledge_extractor.py --json "ocr_results.json" --import-neo4j

  # 使用GPU加速
  python start_knowledge_extractor.py --pdf "textbook.pdf" --use-gpu

  # 分批处理大型文档
  python start_knowledge_extractor.py --pdf "large_doc.pdf" --batch-size 3 --start-index 10 --max-items 20
        """
    )

    # 输入源参数
    parser.add_argument("--pdf", help="PDF文件路径")
    parser.add_argument("--images", help="图片文件夹路径")
    parser.add_argument("--json", help="已提取的文字JSON文件路径")

    # 基本参数
    parser.add_argument("--output", default="knowledge_graph.json",
                        help="输出知识图谱文件名 (默认: knowledge_graph.json)")
    parser.add_argument("--domain", default="计算机科学",
                        help="知识领域 (默认: 计算机科学)")

    # 处理参数
    parser.add_argument("--batch-size", type=int, default=10,
                        help="每批处理的页数/图片数 (默认: 10)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="开始索引，从0开始 (默认: 0)")
    parser.add_argument("--max-items", type=int,
                        help="最大处理数量 (默认: 全部)")

    # OCR参数
    parser.add_argument("--ocr-engine", default="paddle", choices=["paddle", "easyocr"],
                        help="OCR引擎选择 (默认: paddle)")
    parser.add_argument("--ocr-lang", default="ch",
                        help="OCR语言设置 (默认: ch)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="PDF转图像DPI (默认: 300)")

    # 图像保存参数
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="保存PDF转换的图像文件 (默认: 开启)")
    parser.add_argument("--no-save-images", action="store_true",
                        help="不保存图像文件")

    # 模型参数
    parser.add_argument("--model", help="自定义LLM模型路径")
    parser.add_argument("--use-gpu", action="store_true",
                        help="使用GPU加速")

    # Neo4j参数
    parser.add_argument("--import-neo4j", action="store_true",
                        help="导入到Neo4j数据库")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687",
                        help="Neo4j连接URI (默认: bolt://localhost:7687)")
    parser.add_argument("--neo4j-user", default="neo4j",
                        help="Neo4j用户名 (默认: neo4j)")
    parser.add_argument("--neo4j-password", default="admin123",
                        help="Neo4j密码 (默认: admin123)")

    # 其他参数
    parser.add_argument("--resume", action="store_true",
                        help="从中断处恢复处理")
    parser.add_argument("--show-stats", action="store_true",
                        help="显示详细统计信息")

    args = parser.parse_args()

    # 验证输入参数
    if not args.pdf and not args.images and not args.json:
        print("❌ 请指定输入源:")
        print("   --pdf <PDF文件路径>")
        print("   --images <图片文件夹路径>")
        print("   --json <JSON文件路径>")
        parser.print_help()
        return

    # 显示启动信息
    print_banner()

    # 检查环境
    if not check_requirements():
        sys.exit(1)

    # 验证输入文件
    print("\n📁 验证输入文件...")
    if not validate_input_files(args):
        sys.exit(1)

    # 显示处理计划
    show_processing_plan(args)

    # 确认是否继续
    if args.show_stats:
        print("\n" + "=" * 60)
        response = input("是否开始处理? (y/N): ")
        if response.lower() not in ['y', 'yes', '是']:
            print("处理已取消")
            return

    print(f"\n🚀 开始知识图谱提取...")
    print(f"   💡 提示: 处理过程中按 Ctrl+C 可以中断")
    print(f"   📝 日志文件: logs/knowledge_extractor.log")

    try:
        # 准备参数并传递给原始main函数
        original_argv = sys.argv

        # 构建新的argv
        new_argv = ["knowledge_extractor_integrated.py"]

        if args.pdf:
            new_argv.extend(["--pdf", args.pdf])
        elif args.images:
            new_argv.extend(["--images", args.images])
        elif args.json:
            new_argv.extend(["--json", args.json])

        new_argv.extend([
            "--output", args.output,
            "--domain", args.domain,
            "--batch-size", str(args.batch_size),
            "--start-index", str(args.start_index),
            "--ocr-engine", args.ocr_engine,
            "--ocr-lang", args.ocr_lang,
            "--dpi", str(args.dpi)
        ])

        if args.max_items:
            new_argv.extend(["--max-items", str(args.max_items)])
        if args.model:
            new_argv.extend(["--model", args.model])
        if args.use_gpu:
            new_argv.append("--use-gpu")
        if args.import_neo4j:
            new_argv.extend([
                "--import-neo4j",
                "--neo4j-uri", args.neo4j_uri,
                "--neo4j-user", args.neo4j_user,
                "--neo4j-password", args.neo4j_password
            ])
        if args.no_save_images:
            new_argv.append("--no-save-images")
        if args.resume:
            new_argv.append("--resume")
        if args.show_stats:
            new_argv.append("--show-stats")

        # 临时替换sys.argv
        sys.argv = new_argv

        # 调用原始main函数
        extractor_main()

        print(f"\n🎉 知识图谱提取完成！")
        print(f"   📁 输出文件: shared/output/{args.output}")
        if args.import_neo4j:
            print(f"   🗄️  已导入Neo4j数据库")

    except KeyboardInterrupt:
        print("\n\n🛑 收到停止信号...")
        print("👋 知识图谱提取已中断，中间结果已保存")
    except Exception as e:
        print(f"\n❌ 知识图谱提取失败: {e}")
        sys.exit(1)
    finally:
        # 恢复原始argv
        sys.argv = original_argv


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)