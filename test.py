from paddleocr import PaddleOCR

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')
print("PaddleOCR初始化成功")

# 测试识别一张简单图片
result = ocr.ocr('test.png')  # 替换为你的任意图片
print("识别结果:", result)