import casadi
import numpy as np
def vehicle_model(x, u):
    # set physical constants
    l_r = 1.33  # distance rear wheels to center of gravitiy of the car
    l_f = 1.81  # distance front wheels to center of gravitiy of the car

    # set parameters
    beta = casadi.arctan(l_r / (l_f + l_r) * casadi.tan(u[1]))  # 当前的前轮转角决定了速度的侧偏角

    # calculate dx/dt 动力学模型要自己写，一定不要报错！！
    return casadi.vertcat(x[3] * casadi.cos(x[2] + beta),  # dxPos/dt = v*cos(theta+beta)
                          x[3] * casadi.sin(x[2] + beta),  # dyPos/dt = v*sin(theta+beta)
                          x[3] / l_r * casadi.sin(beta),  # dphi/dt=
                          u[0])  # dv/dt = a