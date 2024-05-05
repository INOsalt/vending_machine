import pickle
import numpy as np
import pandas as pd
from collections import deque
import random

# 加载KDE模型
with open('toO.csv_ENDTTIME_kde.pkl', 'rb') as file:
    kde = pickle.load(file)

# 生成一天内的到达时间
def generate_daily_arrivals(kde, num_samples = 133): #1562 50% 133
    # 重新抽样，确保每天的到达时间独立
    index = random.uniform(0.8, 1.2)
    num_samples_random = int(num_samples * index) #* index
    samples = kde.resample(num_samples_random)[0]
    # 将小时转换为分钟
    samples_in_minutes = samples * 60
    # 使用clip确保时间在0到1439分钟之间，这表示一天的开始到结束
    return np.clip(samples_in_minutes, 0, 1439).astype(int)

# 模拟排队和服务时间
def simulate_queue(arrivals, days_in_year):
    minutes_in_day = 1440
    year_usage = []
    unattended_customers = 0  # 添加未接待客户计数器

    for day in range(days_in_year):
        daily_usage = np.zeros(minutes_in_day)
        queue = deque()
        currently_serving = None
        service_end_time = -1

        for minute in range(minutes_in_day):
            # 检查当前服务是否完成
            if currently_serving is not None and minute >= service_end_time:
                currently_serving = None

            # 处理当前分钟的到达
            arrivals_now = np.sum(arrivals[day] == minute)
            for _ in range(arrivals_now):
                queue.append((minute, np.random.randint(2, 6)))  # 加入队列并随机分配服务时间 1- 3min

            # 如果当前没有客户在接受服务且队列中有等待的客户，从队列中取出一个客户开始服务
            if currently_serving is None and queue:
                start_time, service_time = queue.popleft()
                service_end_time = minute + service_time
                # 标记使用时间
                if service_end_time < minutes_in_day:
                    for t in range(minute, min(service_end_time, minutes_in_day)):
                        daily_usage[t] = 1
                currently_serving = minute

        # 在每天结束时计算队列中剩余的客户
        unattended_customers += len(queue)

        year_usage.append(daily_usage)

    return np.array(year_usage).flatten(), unattended_customers

# 为一整年生成到达时间，每天重新抽样
days_in_year = 365
daily_arrivals = [generate_daily_arrivals(kde) for _ in range(days_in_year)]

# 将daily_arrivals转换为DataFrame
arrivals_df = pd.DataFrame(daily_arrivals)
# 转置DataFrame以便每列代表一天的到达时间
arrivals_df = arrivals_df.transpose()
# 保存到CSV文件
arrivals_df.to_csv('daily_arrivals.csv', index=False)
print("Daily arrivals data has been saved to 'daily_arrivals.csv'.")

# 应用排队模型并获取未接待的客户数量
yearly_usage, total_unattended = simulate_queue(daily_arrivals, days_in_year)

# 打印未接待客户的结果
print(f"Total unattended customers: {total_unattended}")

# 统计1的数量
ones_count = np.count_nonzero(yearly_usage)
# 统计0的数量
zeros_count = len(yearly_usage) - ones_count
# 打印结果
print(f"Number of 1's (Machine Used): {ones_count}")
print(f"Number of 0's (Machine Not Used): {zeros_count}")

# 转换为DataFrame
yearly_data = pd.DataFrame(yearly_usage, columns=['Machine Used'])
# 保存到CSV文件
yearly_data.to_csv('yearly_vending_machine_usage.csv', index=False)
print("Data saved to 'yearly_vending_machine_usage.csv'.")

def process_vending_machine_usage(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 检查列名，确保适用性
    if df.columns.size != 1:
        raise ValueError("CSV文件应只包含一列")
    # 获取数据列的名字
    interaction_column = df.columns[0]
    # 初始化结果列表，用0填充与输入数据等长的列表
    working = [0] * len(df)
    # 处理售货机工作状态
    i = 0
    while i < len(df):
        if df[interaction_column][i] == 1:
            # 如果检测到互动，设置接下来的五分钟或直到数据末尾的状态为1
            for j in range(i, min(i + 5, len(df))):
                working[j] = 1
            i += 5  # 跳过接下来的五分钟，因为机器将工作这段时间
        else:
            i += 1  # 如果没有互动，继续检查下一分钟

    # 统计1的数量
    ones_count = np.count_nonzero(working)
    # 统计0的数量
    zeros_count = len(working) - ones_count
    # 打印结果
    print(f"Number of 1's (Machine Used): {ones_count}")
    print(f"Number of 0's (Machine Not Used): {zeros_count}")

    # 将结果列表转换为DataFrame
    result_df = pd.DataFrame(working, columns=['vending_machine'])

    # 输出结果到新的CSV文件
    result_df.to_csv('yearly_vending_machine_activity.csv', index=False)

    print("处理完成，结果已输出到 'yearly_vending_machine_activity.csv'")

# 固定时间
process_vending_machine_usage('yearly_vending_machine_usage.csv')
