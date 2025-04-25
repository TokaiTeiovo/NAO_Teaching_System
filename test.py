import paddle

print(f"PaddlePaddle版本: {paddle.__version__}")

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ch')
print("PaddleOCR初始化成功")