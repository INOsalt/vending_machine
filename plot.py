import pandas as pd
import matplotlib.pyplot as plt

def weather():
    # 读取CSV文件
    df = pd.read_csv('weather.csv')

    # 检查需要的列是否存在
    if {'Dry', 'Total_radiation'}.issubset(df.columns):
        # 计算每120个数据的平均值
        df_mean = df[['Dry', 'Total_radiation']].rolling(window=120).mean().dropna()

        # 创建一个图形框架，设置图的大小
        fig, ax1 = plt.subplots(figsize=(15, 8))

        # 创建第一个y轴，显示总辐射
        color = 'tab:red'
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Total Solar Radiation (W/m²)', color=color)
        ax1.plot(df_mean.index / 60, df_mean['Total_radiation'], label='Total Solar Radiation', color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)

        # 创建第二个y轴，显示温度
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Temperature (°C)', color=color)
        ax2.plot(df_mean.index / 60, df_mean['Dry'], label='Temperature', color=color, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)

        # 添加图的其他元素
        plt.title('Weather Data (Averaged Every 2h)')
        fig.tight_layout()  # 调整整体布局
        plt.grid(True)
        # 显示图例
        plt.legend()
        plt.show()
    else:
        print("Some required columns are missing in the CSV file.")

def result_plot(file):
    # 读取CSV文件
    df = pd.read_csv(file)

    # 检查需要的列是否存在
    if {'Surplus2nd_kJph', 'Short2nd_kJph'}.issubset(df.columns):
        # 将kJph转换为W，通过除以3.6
        df['Surplus2nd_W'] = df['Surplus2nd_kJph'] / 3.6
        df['Short2nd_W'] = df['Short2nd_kJph'] / 3.6

        # 创建一个图形框架，设置图的大小
        fig, ax = plt.subplots(figsize=(15, 8))

        # 绘制第一个数据集 - 进口功率
        color = 'tab:red'
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power (W)')
        ax.plot(df.index / 60, df['Surplus2nd_W'], label='Import Power', color=color, alpha=0.7, linewidth=1)
        ax.tick_params(axis='y', labelcolor=color)

        # 绘制第二个数据集 - 减产
        color = '#003f5c'  # 青绿色
        ax.plot(df.index / 60, df['Short2nd_W'], label='Curtailment', color=color, alpha=0.7, linewidth=1)

        # 添加图的其他元素
        plt.title('Base case (current)')
        fig.tight_layout()  # 调整整体布局
        plt.grid(True)
        # 显示图例
        plt.legend()
        plt.show()
    else:
        print("Some required columns are missing in the CSV file.")

def SOC_plot(file):
    # 读取CSV文件
    df = pd.read_csv(file)

    # 检查需要的列是否存在
    if {'FSOC'}.issubset(df.columns):

        # 创建一个图形框架，设置图的大小
        fig, ax = plt.subplots(figsize=(15, 8))

        color = '#6a0dad'
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('SOC')
        ax.plot(df.index / 60, df['FSOC'], label='Battery SOC', color=color, alpha=0.7, linewidth=1)
        ax.tick_params(axis='y', labelcolor=color)

        # 设置y轴的范围为0到1
        ax.set_ylim(0, 1)

        # 添加图的其他元素
        plt.title('Base case (current)')
        fig.tight_layout()  # 调整整体布局
        plt.grid(True)
        # 显示图例
        plt.legend()
        plt.show()
    else:
        print("Some required columns are missing in the CSV file.")



def com_plot(file1,file2):
    # 读取CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 检查需要的列是否存在
    if {'Surplus2nd_kJph', 'Short2nd_kJph'}.issubset(df1.columns):
        # 将kJph转换为W，通过除以3.6
        df1['Surplus2nd_W'] = df1['Surplus2nd_kJph'] / 3.6
        df1['Short2nd_W'] = df1['Short2nd_kJph'] / 3.6

    if {'Surplus2nd_kJph', 'Short2nd_kJph'}.issubset(df1.columns):
        # 将kJph转换为W，通过除以3.6
        df2['Surplus2nd_W'] = df2['Surplus2nd_kJph'] / 3.6
        df2['Short2nd_W'] = df2['Short2nd_kJph'] / 3.6

        # 创建一个图形框架，设置图的大小
        fig, ax = plt.subplots(figsize=(15, 8))

        # 绘制第一个数据集 - 进口功率
        color = '#2c3e50'
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Power (W)')
        ax.plot(df1.index / 60, df1['Short2nd_W'], label='Case1', color=color, alpha=0.8, linewidth=1)
        ax.tick_params(axis='y', labelcolor=color)

        # 绘制第二个数据集 - 减产
        color = '#ff7f50'  # 青绿色
        ax.plot(df1.index / 60, df2['Short2nd_W'], label='Case3', color=color, alpha=0.8, linewidth=1, linestyle='--')

        # 添加图的其他元素
        plt.title('Import Power')
        fig.tight_layout()  # 调整整体布局
        plt.grid(True)
        # 显示图例
        plt.legend()
        plt.show()
    else:
        print("Some required columns are missing in the CSV file.")

def comSOC_plot(file1,file2):
    # 读取CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 检查需要的列是否存在
    if {'FSOC'}.issubset(df1.columns):

        # 创建一个图形框架，设置图的大小
        fig, ax = plt.subplots(figsize=(15, 8))

        color = '#6a0dad'
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('SOC')
        ax.plot(df1.index / 60, df1['FSOC'], label='Case2 ', color=color, alpha=0.7, linewidth=1, linestyle='--')
        ax.tick_params(axis='y', labelcolor=color)

        # 设置y轴的范围为0到1
        ax.set_ylim(0, 1)

        # 绘制第二个数据集 - 减产
        color = '#ff7f50'
        ax.plot(df1.index / 60, df2['FSOC'], label='Case4', color=color, alpha=0.8, linewidth=1)

        # 添加图的其他元素
        plt.title('Battery SOC')
        fig.tight_layout()  # 调整整体布局
        plt.grid(True)
        # 显示图例
        plt.legend()
        plt.show()
    else:
        print("Some required columns are missing in the CSV file.")


file = 'RESULT/5130-BASE.csv'
result_plot(file)
SOC_plot(file)

# file1 = 'RESULT/5130-BASE10day.csv'
# file2 = 'RESULT/5130-MPC-con10day.csv'

# file1 = 'RESULT/5130-5min10day.csv'
# file2 = 'RESULT/5130-MPC-10day.csv'

# com_plot(file1,file2)
# comSOC_plot(file1,file2)