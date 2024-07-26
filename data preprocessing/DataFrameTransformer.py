import pandas as pd

class DataFrameTransformer:
    def __init__(self, dataframe):
        """
        初始化类
        :param dataframe: pandas DataFrame，待处理的数据框
        """
        self.dataframe = dataframe

    def str_to_num(self, column):
        """
        将指定列中的 'num' 字符串转化为数字
        :param column: str，待处理的列名
        :return: 处理后的 DataFrame
        """
        self.dataframe[column] = self.dataframe[column].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
        return self.dataframe

    def num_to_str(self, column):
        """
        将指定列中的数字转化为 'num' 字符串
        :param column: str，待处理的列名
        :return: 处理后的 DataFrame
        """
        self.dataframe[column] = self.dataframe[column].apply(lambda x: str(x) if isinstance(x, int) else x)
        return self.dataframe

    def transpose_matrix(self):
        """
        将 DataFrame 进行转置
        :return: 转置后的 DataFrame
        """
        return self.dataframe.T

# 示例用法
if __name__ == "__main__":
    # 创建示例 DataFrame
    data = {
        'num': ['1', '2', '3', '4', '5'],
        'value': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    transformer = DataFrameTransformer(df)
    
    # 将 'num' 转化为数字
    df_num = transformer.str_to_num('num')
    print("将 'num' 转化为数字:\n", df_num)

    # 将数字转化为 'num' 字符串
    df_str = transformer.num_to_str('num')
    print("将数字转化为 'num' 字符串:\n", df_str)

    # 转置矩阵
    df_transposed = transformer.transpose_matrix()
    print("转置矩阵:\n", df_transposed)
