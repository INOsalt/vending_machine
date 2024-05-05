from docplex.mp.model import Model
import numpy as np
import pandas as pd
import math
import pickle
from scipy.integrate import quad
import time

def calculate_machine_usage(minute):
    # 加载KDE模型
    with open('toO_kde.pkl', 'rb') as file:
        kde = pickle.load(file)
    integrated_probabilities = quad(kde, minute/60, (minute + 1)/60)[0]
    # 假设每个人使用5分钟
    time_use = integrated_probabilities * 133 * 5
    return time_use


def calculate_vending_machine_load(total_minutes):
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



# 读取CSV文件中的气象数据
def load_weather_data(filename, time_start):
    df = pd.read_csv(filename)
    temperature = df.loc[time_start:time_start+59, 'Dry'].to_numpy()
    solar_zenith_angle = df.loc[time_start:time_start+59, 'Solar_zenith'].to_numpy()
    irradiance = df.loc[time_start:time_start+59, 'Total_radiation'].to_numpy()
    return temperature, solar_zenith_angle, irradiance
def calculate_pv_output(temperature_celsius, solar_zenith_angle, irradiance, panel_tilt=20, panel_area=1.6, efficiency=18.5):
    # 常量
    STC_Irradiance = 1000  # 标准测试条件下的辐照强度 (W/m^2)
    Voc_STC = 40.7  # 开路电压, 实际值 (V)
    Isc_STC = 11.24  # 短路电流, 实际值 (A)
    Temp_Coeff_Voc = -0.25 / 100  # 开路电压的温度系数 (%/°C)
    Temp_Coeff_Isc = 0.04 / 100  # 短路电流的温度系数 (%/°C)
    Nominal_Power = 360  # 标称功率 (W)
    Temp_Coeff_Power = -0.34 / 100  # 最大功率的温度系数 (%/°C)
    n = 1.3  # 理想因子
    k = 1.38e-23  # 玻尔兹曼常数
    q = 1.602e-19  # 元电荷
    T_kelvin = temperature_celsius + 273.15  # 将输入的摄氏度转换为开尔文

    # 计算热电压
    Vt = n * k * T_kelvin / q

    # 温度对Voc和Isc的影响
    Voc = Voc_STC * (1 + Temp_Coeff_Voc * (temperature_celsius - 25))
    Isc = Isc_STC * (1 + Temp_Coeff_Isc * (temperature_celsius - 25))

    # 计算反向饱和电流
    exponent = Voc / (n * Vt)
    exponent = min(exponent, 709)  # 确认避免溢出
    I0 = Isc / (np.exp(exponent) - 1)

    # 使用单二极管模型计算实际电流，同样考虑溢出问题
    exponent = 12 / (n * Vt)
    exponent = min(exponent, 709)  # 再次确认避免溢出
    I = Isc - I0 * (np.exp(exponent) - 1)

    # 计算实际接收的辐照强度（考虑光伏板角度和天顶角）
    effective_irradiance = irradiance / 3.6 * math.cos(math.radians(abs(solar_zenith_angle - panel_tilt)))

    # 调整最大功率根据面积和效率
    adjusted_power = Nominal_Power * (effective_irradiance / STC_Irradiance) * (panel_area * (efficiency / 100))
    adjusted_power *= (1 + Temp_Coeff_Power * (temperature_celsius - 25))

    # PWM控制器输出电压调整到12V，计算输出功率
    if Voc_STC > 12:  # 确保有足够电压以供PWM控制器工作
        output_power = min(adjusted_power, I * 12)  # 使用计算的电流和12V计算输出功率
    else:
        output_power = min(adjusted_power, I * Voc)  # 使用计算的电流和Voc计算输出功率

    return output_power

def MPC(start_time,step):
    # 参数和变量定义
    time_horizon = 60
    max_charge_rate = 120
    max_discharge_rate = 120
    battery_capacity = 144*2
    initial_soc = 50

    # 创建docplex模型
    mdl = Model()

    # 定义决策变量
    charge_rate = mdl.continuous_var_list(time_horizon, lb=0, ub=max_charge_rate, name='charge_rate')
    discharge_rate = mdl.continuous_var_list(time_horizon, lb=0, ub=max_discharge_rate, name='discharge_rate')
    grid_import = mdl.continuous_var_list(time_horizon, lb=0, name='grid_import')

    # 加载气象数据
    temperature, solar_zenith_angle, irradiance = load_weather_data('weather.csv', start_time)
    temperature_mean = temperature.reshape(-1, 10).mean(axis=1)
    solar_zenith_angle_mean = solar_zenith_angle.reshape(-1, 10).mean(axis=1)
    irradiance_mean = irradiance.reshape(-1, 10).mean(axis=1)

    # 创建约束
    soc = initial_soc
    machine_usage = calculate_vending_machine_load(time_horizon)
    machine_usage_sum = irradiance.reshape(-1, 10).sum(axis=1)
    print(machine_usage)
    for t in range(time_horizon):
        pv_output = calculate_pv_output(temperature[t], solar_zenith_angle[t], irradiance[t])

        # 更新SOC
        soc = soc + charge_rate[t] - discharge_rate[t]
        # SOC约束
        mdl.add_constraint(soc >= 20)
        mdl.add_constraint(soc <= battery_capacity)
        # 电网购买电量约束
        net_power_usage = machine_usage[t] - pv_output + charge_rate[t] - discharge_rate[t]
        mdl.add_constraint(grid_import[t] == mdl.max(0, net_power_usage))

    # 定义目标函数
    mdl.minimize(mdl.sum(grid_import))

    # 求解模型
    sol = mdl.solve()
    charge = sol[charge_rate[0]]
    discharge = sol[charge_rate[0]]
    net_battery = (charge - discharge) * step # W

    return net_battery

start = time.time()
MPC(0,1)
end = time.time()
print (end-start)