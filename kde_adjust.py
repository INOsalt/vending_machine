import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import seaborn as sns
from collections import deque

# 加载KDE模型
with open('toO_kde.pkl', 'rb') as file:
    kde = pickle.load(file)

# 设置总天数
total_days = 1000

# 一天的总分钟数
minutes_in_day = 1440

# 初始化服务时间记录
all_service_minutes = np.zeros(minutes_in_day)

def generate_daily_arrivals(kde, num_samples = 133): #1562 50% 133
    # 重新抽样，确保每天的到达时间独立
    num_samples_random = num_samples #* index
    samples = kde.resample(num_samples_random)[0]
    # 将小时转换为分钟
    samples_in_minutes = samples * 60
    # 使用clip确保时间在0到1439分钟之间，这表示一天的开始到结束
    return np.clip(samples_in_minutes, 0, 1439).astype(int)

arrivals = [generate_daily_arrivals(kde) for _ in range(total_days)]

# 循环多天
for day in range(total_days):
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
            service_end_time = start_time + service_time
            # 标记使用时间
            if service_end_time < minutes_in_day:
                for t in range(start_time, min(service_end_time, minutes_in_day)):
                    daily_usage[t] = 1
            currently_serving = start_time

    # 将当日服务时间记录到总记录中
    # Aggregate daily usage into total usage count for each minute
    all_service_minutes += daily_usage

# Fit a new gaussian_kde model to the non-zero data points
data_points = np.repeat(np.arange(minutes_in_day), all_service_minutes.astype(int))
kde_service = gaussian_kde(data_points)

# Generate values for the fitted KDE
x_d = np.linspace(0, 1440, 500)
density = kde_service(x_d)

plt.figure(figsize=(10, 5))
sns.histplot(data_points, kde=False, bins=30, color='skyblue', stat="density")
# ax.hist(data_points, bins=48, density=True, alpha=0.5, label='Histogram of Usage')
plt.plot(x_d, density, label='KDE Fit', color='red')
plt.title(f'KDE for Vending Machine Usage Frequencies')
plt.xlabel('Time of Day (minutes)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Save the new KDE model
with open('new_usage_kde.pkl', 'wb') as file:
    pickle.dump(kde_service, file)