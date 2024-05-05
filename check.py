import pickle
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
# Re-attempting to load the gaussian_kde object from 'new_usage_kde.pkl' and perform integration per minute
# with open('new_usage_kde.pkl', 'rb') as file:
#     loaded_kde = pickle.load(file)
#
# # Defining function to integrate KDE from 'start' to 'end' minutes
# def integrate_per_minute(kde, start, end):
#     # Convert minutes to hours for kde which was fitted with hours
#     return quad(kde, start, end)[0]

# Array to hold the integrated probabilities for each minute
# integrated_probabilities = np.array([integrate_per_minute(loaded_kde, minute, minute + 1) for minute in range(1440)])
# minute = 800
# integrated_probabilities = integrate_per_minute(loaded_kde, minute, minute + 1)
# # Show the first 10 integrated probabilities
# print(integrated_probabilities)

def calculate_machine_usage(minute):
    # 加载KDE模型
    with open('toO_kde.pkl', 'rb') as file:
        kde = pickle.load(file)
    integrated_probabilities = quad(kde, minute/60, (minute + 1)/60)[0]
    # 假设每个人使用5分钟
    time_use = integrated_probabilities * 133 * 5
    return time_use


def calculate_vending_machine_load():
    # 假定总分钟数
    total_minutes = 60
    # 售货机每分钟的负载
    vending_machine_load = np.zeros(total_minutes)

    # 累积时间对应的结束时间计算
    current_time = 0
    for minute in range(total_minutes):
        usage = calculate_machine_usage(minute)
        # 确保duration是整数
        duration = int(round(usage))  # 对使用时间进行四舍五入并转换为整数
        end_time = minute + duration  # 结束时间为当前分钟加上持续时间

        # 标记售货机工作的分钟
        for active_minute in range(minute, min(end_time, total_minutes)):
            vending_machine_load[active_minute] = 15  # 售货机在运行时的功率为15W

    return vending_machine_load


# 使用示例
vending_machine_load = calculate_vending_machine_load()
print("每分钟售货机负载（W）:", vending_machine_load)


