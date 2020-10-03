from data_processor import DataProcessor
from word_segmentor import WordSegmentor

if __name__ == "__main__":
    # 数据处理
    dp = DataProcessor()
    dp.read()
    dp.build_dict()
    dp.get_text()

    # 分词 + 评价
    ws = WordSegmentor()
    ws.segment()
    ws.evaluate()