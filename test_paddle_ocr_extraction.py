# test_pure_paddle_ocr.py
"""
测试纯PaddlePaddle OCR实现，不依赖PyTorch
"""

import os
import sys
import argparse
import time
import paddle


def verify_paddle_installation():
    """验证PaddlePaddle安装是否正确"""
    try:
        # 检查PaddlePaddle版本
        print(f"PaddlePaddle版本: {paddle.__version__}")

        # 检查GPU支持
        if paddle.device.is_compiled_with_cuda():
            print("PaddlePaddle已编译支持CUDA")
            gpu_count = paddle.device.cuda.device_count()
            print(f"可用GPU数量: {gpu_count}")
        else:
            print("PaddlePaddle仅支持CPU")

        print("PaddlePaddle安装验证成功!")
        return True
    except Exception as e:
        print(f"PaddlePaddle安装验证失败: {e}")
        return False


def test_pure_paddle_ocr(pdf_path, sample_pages=5, lang='ch', use_gpu=False):
    """测试纯PaddlePaddle OCR提取器"""
    try:
        # 导入纯PaddlePaddle OCR提取器
        from knowledge_extraction.pure_paddle_ocr_extractor import PurePaddleOCRPDFExtractor

        print(f"测试文件: {pdf_path}")
        print(f"语言: {lang}")
        print(f"使用GPU: {'是' if use_gpu else '否'}")
        print(f"处理页数: {sample_pages}")

        # 创建OCR提取器
        start_time = time.time()
        extractor = PurePaddleOCRPDFExtractor(pdf_path, lang=lang, use_gpu=use_gpu)
        init_time = time.time() - start_time
        print(f"初始化耗时: {init_time:.2f}秒")

        # 提取文本
        start_time = time.time()
        text = extractor.extract_sample(num_pages=sample_pages)
        extract_time = time.time() - start_time
        print(f"提取文本耗时: {extract_time:.2f}秒")

        # 保存结果
        if text:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            output_file = os.path.join(temp_dir, "pure_paddle_ocr_test.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"提取结果已保存到: {output_file}")
            print(f"文本长度: {len(text)}字符")

            # 显示部分结果
            print("\n提取的文本样本(前200字符):")
            print(text[:200])
        else:
            print("未能提取任何文本")

        return True
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试纯PaddlePaddle OCR")
    parser.add_argument("--pdf", required=True, help="PDF文件路径")
    parser.add_argument("--pages", type=int, default=5, help="测试页数")
    parser.add_argument("--lang", default="ch", help="OCR语言(ch/en)")
    parser.add_argument("--use_gpu", action="store_true", help="使用GPU加速")

    args = parser.parse_args()

    print("=" * 50)
    print("纯PaddlePaddle OCR测试")
    print("=" * 50)

    if verify_paddle_installation():
        print("\n开始测试纯PaddlePaddle OCR...")
        result = test_pure_paddle_ocr(
            args.pdf,
            sample_pages=args.pages,
            lang=args.lang,
            use_gpu=args.use_gpu
        )

        if result:
            print("\n测试完成！纯PaddlePaddle OCR工作正常。")
        else:
            print("\n测试失败！请检查错误信息。")
    else:
        print("\nPaddlePaddle安装验证失败，无法进行测试。")
        print("请先安装正确版本的PaddlePaddle。")

    print("=" * 50)