# Python module for the TRNSYS Type calling Python using CFFI
# Data exchange with TRNSYS uses a dictionary, called TRNData in this file (it is the argument of all functions).
# Data for this module will be in a nested dictionary under the module name,
# i.e. if this file is called "MyScript.py", the inputs will be in TRNData["MyScript"]["inputs"]
# for convenience the module name is saved in thisModule
#
# MKu, 2022-02-15

import numpy
import os
from docplex.mp.model import Model
import pandas as pd
import math
import pickle
from scipy.integrate import quad

thisModule = os.path.splitext(os.path.basename(__file__))[0]

# Initialization: function called at TRNSYS initialization
# ----------------------------------------------------------------------------------------------------------------------
def Initialization(TRNData):

    # This model has nothing to initialize
    
    return


# StartTime: function called at TRNSYS starting time (not an actual time step, initial values should be reported)
# ----------------------------------------------------------------------------------------------------------------------
def StartTime(TRNData):

    # Define local short names for convenience (this is optional)
    start_time = 0
    step = TRNData[thisModule]["inputs"][1]
    initial_soc = TRNData[thisModule]["inputs"][2]
    pv_output = TRNData[thisModule]["inputs"][3]
    machine_use0 = TRNData[thisModule]["inputs"][4]

    # Calculate the outputs
    net_charge,net_discharge = MPC(start_time, step,initial_soc,pv_output,machine_use0)

    # Set outputs in TRNData
    TRNData[thisModule]["outputs"][0] = 0
    TRNData[thisModule]["outputs"][1] = 0

    return


# Iteration: function called at each TRNSYS iteration within a time step
# ----------------------------------------------------------------------------------------------------------------------
def Iteration(TRNData):

    # Define local short names for convenience (this is optional)
    start_time = TRNData[thisModule]["inputs"][0]
    step = TRNData[thisModule]["inputs"][1]
    initial_soc = TRNData[thisModule]["inputs"][2]
    pv_output = TRNData[thisModule]["inputs"][3]
    machine_use0 = TRNData[thisModule]["inputs"][4]

    # Calculate the outputs
    net_charge,net_discharge = MPC(start_time, step,initial_soc,pv_output,machine_use0)

    # Set outputs in TRNData
    TRNData[thisModule]["outputs"][0] = net_charge
    TRNData[thisModule]["outputs"][1] = net_discharge
   
    return

# EndOfTimeStep: function called at the end of each time step, after iteration and before moving on to next time step
# ----------------------------------------------------------------------------------------------------------------------
def EndOfTimeStep(TRNData):

    # This model has nothing to do during the end-of-step call
    
    return


# LastCallOfSimulation: function called at the end of the simulation (once) - outputs are meaningless at this call
# ----------------------------------------------------------------------------------------------------------------------
def LastCallOfSimulation(TRNData):

    # NOTE: TRNSYS performs this call AFTER the executable (the online plotter if there is one) is closed. 
    # Python errors in this function will be difficult (or impossible) to diagnose as they will produce no message.
    # A recommended alternative for "end of simulation" actions it to implement them in the EndOfTimeStep() part, 
    # within a condition that the last time step has been reached.
    #
    # Example (to be placed in EndOfTimeStep()):
    #
    # stepNo = TRNData[thisModule]["current time step number"]
    # nSteps = TRNData[thisModule]["total number of time steps"]
    # if stepNo == nSteps-1:     # Remember: TRNSYS steps go from 0 to (number of steps - 1)
    #     do stuff that needs to be done only at the end of simulation

    return


def calculate_machine_usage(minute):
    # 加载KDE模型
    with open('toO_kde.pkl', 'rb') as file:
        kde = pickle.load(file)
    integrated_probabilities = quad(kde, minute / 60, (minute + 1) / 60)[0]
    # 假设每个人使用5分钟
    time_use = integrated_probabilities * 133 * 5
    return time_use


def calculate_vending_machine_load(total_minutes,start_time):
    # 售货机每分钟的负载
    vending_machine_load = numpy.zeros(total_minutes)
    # 累积时间对应的结束时间计算
    current_time = 0
    for minute in range(start_time, total_minutes):
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
    temperature = df.loc[time_start:time_start + 59, 'Dry'].to_numpy()
    solar_zenith_angle = df.loc[time_start:time_start + 59, 'Solar_zenith'].to_numpy()
    irradiance = df.loc[time_start:time_start + 59, 'Total_radiation'].to_numpy()
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
    I0 = Isc / (numpy.exp(exponent) - 1)

    # 使用单二极管模型计算实际电流，同样考虑溢出问题
    exponent = 12 / (n * Vt)
    exponent = min(exponent, 709)  # 再次确认避免溢出
    I = Isc - I0 * (numpy.exp(exponent) - 1)

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


def MPC(start_time, step,initial_soc,pv_output0,machine_use0):
    # 参数和变量定义
    start_time = int(start_time)
    time_horizon = 30
    max_charge_rate = 120
    max_discharge_rate = 120
    battery_capacity = 144 * 2

    # 创建docplex模型
    mdl = Model()

    # 定义决策变量
    charge_rate = mdl.continuous_var_list(time_horizon, lb=0, ub=max_charge_rate, name='charge_rate')
    discharge_rate = mdl.continuous_var_list(time_horizon, lb=0, ub=max_discharge_rate, name='discharge_rate')
    grid_import = mdl.continuous_var_list(time_horizon, lb=0, name='grid_import')

    # 加载气象数据
    temperature, solar_zenith_angle, irradiance = load_weather_data('weather.csv', start_time)

    # 创建约束
    soc = initial_soc
    machine_usage = calculate_vending_machine_load(time_horizon,start_time)
    print(machine_usage)
    for t in range(time_horizon):
        if t == 0:
            pv_output = pv_output0
            machine_use = machine_use0
        else:
            pv_output = calculate_pv_output(temperature[t], solar_zenith_angle[t], irradiance[t])
            machine_use = machine_usage[t]

        # 更新SOC
        soc = soc + (charge_rate[t] - discharge_rate[t])/battery_capacity
        # SOC约束
        mdl.add_constraint(soc >= 0.1)
        mdl.add_constraint(soc <= 0.9)
        # 电网购买电量约束
        net_power_usage = machine_use - pv_output + charge_rate[t] - discharge_rate[t]
        mdl.add_constraint(grid_import[t] == mdl.max(0, net_power_usage))

    # 定义目标函数
    mdl.minimize(mdl.sum(grid_import))

    # 求解模型
    sol = mdl.solve()
    charge = sol[charge_rate[0]]
    discharge = sol[charge_rate[0]]
    net_charge = charge * 3.6  # kJ/h
    net_discharge = discharge * 3.6  # kJ/h

    return net_charge,net_discharge
