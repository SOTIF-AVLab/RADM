import numpy as np
import forcespro
import os
import sys
import casadi
import pandas as pd
from scipy.interpolate import CubicSpline
import math
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.patches import Ellipse, Circle
import matplotlib.gridspec as gridspec
import time

solver = forcespro.nlp.Solver.from_directory("/home/kai/Vscode/RL_Decision_Making/src/MPCC/FORCESNLPsolver")  # load solver

# to define the dynamics of car, use for update the state
# u=(a,delta,v_s,s_v)  x=(x,y,phi,v,theta)
# def continuous_dynamics(x, u):
#     # set physical constants
#     l_r = 1.33  # distance rear wheels to center of gravitiy of the car
#     l_f = 1.81  # distance front wheels to center of gravitiy of the car

#     # set parameters
#     beta = casadi.arctan(l_r / (l_f + l_r) * casadi.tan(u[1]))  # 当前的前轮转角决定了速度的侧偏角

#     # calculate dx/dt 动力学模型要自己写，一定不要报错！！
#     return casadi.vertcat(x[3] * casadi.cos(x[2] + beta),  # dxPos/dt = v*cos(theta+beta)
#                           x[3] * casadi.sin(x[2] + beta),  # dyPos/dt = v*sin(theta+beta)
#                           x[3] / l_r * casadi.sin(beta),  # dphi/dt=
#                           u[0],
#                           u[2])  # dv/dt = a


# # use an explicit RK4 integrator here to discretize continuous dynamics to updata state
# integrator_stepsize = 0.1
# equation = lambda z: forcespro.nlp.integrate(continuous_dynamics, z[4:9], z[0:4],
#                                              integrator=forcespro.nlp.integrators.RK4,
#                                              stepsize=integrator_stepsize)

# # two utility functions
# deg2rad = lambda deg: deg/180*math.pi
# rad2deg = lambda rad: rad/math.pi*180


class MPCC:
    def __init__(self, current_state_):  # 传递进来一个初始化的状态
        my_track = pd.read_csv("/home/kai/Vscode/RL_Decision_Making/src/MPCC/track.csv")
        # x_ref and y_ref is all waypoints
        self.x_ref = my_track.loc[0].values  # convet to the numpy ndarray
        
        self.y_ref = my_track.loc[1].values
        
        self.waypoints_size = self.x_ref.size  # total number of waypoints
        
        self.n_pts = 20  # we need 20 points to represent the whole path

        
        # the value of 20 points
        self.ss = np.zeros([self.n_pts])  # 20个点，但是是19组多项式系数  这个是从0开始的
        self.xx = np.zeros([self.n_pts])
        self.yy = np.zeros([self.n_pts])
        # to represent path
        self.final_ref_path_x, self.final_ref_path_y = self.ref_path()
        
        self.current_state = current_state_  # (x,y,phi,v)

        # the MPCC parameters
        self.solver_N = 20
        self.solver_NU = 4  # s.t. (a,delta,v_virtual,slack_value)
        self.solver_NX = 5  # s.t. (x,y,phi,v,arc_length)
        self.solver_total_V = 9
        self.solver_NPAR = 119  # all parameters received by solver

        x0i = np.array([0., 0., 0., 0., 1.6, -200.0, np.pi/2, 1, 0], dtype="double")  # 数据精确度要设置的足够高才可以

        self.solver_x0 = np.transpose(np.tile(x0i, (1, self.solver_N)))
        # the input value to solver
        # 后期在初始x0上指定一个初始值(待完成)
        # self.solver_x0 = np.zeros(self.solver_N*self.solver_total_V)  # the x0 to solve problem
        self.solver_xinit = np.zeros(self.solver_NX, dtype="double")  # the initial state of MPCC
        self.solver_all_parameters = np.zeros(self.solver_NPAR * self.solver_N, dtype="double")  # the runtime parameters

        # weights
        self.velocity_weight = 4
        self.error_contour = 130
        self.error_lag = 130

        self.acceleration_weight = 6
        self.delta_weight = 100
        self.slack_weight = 2800  # 松弛变量的权重取得较大一些
        self.Vs_weight = 0.1  # v_virtual(s.t.,velocity along the arc)

        # the velocity we want to keep
        self.reference_velocity = 10
        self.traj_i = 0  # 当前点最近的s值所在的多项式曲线的下限点

        # plot
        # self.figure = plt.figure(figsize=(9, 9))

        # flag
        self.flag = 0  # 运行run_solver的次数,每运行一次，次数加1
        self.stop_step =0

        # # obstalce
        # self.ObstOne_x = 1.6      # 69      -10
        # self.ObstOne_y = -10      # -10      20
        # self.ObstOne_heading = -deg2rad(0)

        # self.ObstTwo_x = 1.6        # 40        7
        # self.ObstTwo_y = -53.5-50    # -53.5    35
        # self.ObstTwo_heading = -deg2rad(0)

    def ref_path(self):
        x_coordinate = self.x_ref
        y_coordinate = self.y_ref
        waypoints_size = self.waypoints_size  # int
        # way 1: to integrate (通过积分)
        # to find more point to make approximation more accurate
        t = np.arange(0, waypoints_size)  # [0,1,2,...,99], total number is 100
        cs_x = CubicSpline(t, x_coordinate)  # class
        cs_y = CubicSpline(t, y_coordinate)  # class
        de_x = cs_x.derivative()  # calculate the derivative
        de_y = cs_y.derivative()
        u_min = cs_x.x[:-1]  # 每一段进行积分的下限 the size is 99
        u_max = cs_x.x[1:]  # 每一段进行积分的上限
        all_si = np.zeros([u_min.size])

        for i in range(u_min.size):
            f = lambda c: np.sqrt(de_x(c) ** 2 + de_y(c) ** 2)  # the function about the const value c
            si = scipy.integrate.quad(f, u_min[i], u_max[i])  # quad函数返回两个值，第一个值是积分的值，第二个值是对积分值的绝对误差估计。
            all_si[i] = si[0]
        # the consum arc length
        s = np.append([0], np.cumsum(all_si))  # the size is 100
        total_length = s[-1]  # 111.0164
        # then we build our spline
        s_x = CubicSpline(s, x_coordinate)  # 这里是较为密集的插值
        s_y = CubicSpline(s, y_coordinate)
        dist_spline_pts = total_length / (self.n_pts - 1)

        # 将密集的插值离散化
        for i in range(self.n_pts):  # (0,1,...19) total 20
            self.ss[i] = dist_spline_pts * i
            self.xx[i] = s_x(self.ss[i])
            self.yy[i] = s_y(self.ss[i])
        final_x = CubicSpline(self.ss, self.xx)  # generated path_x
        final_y = CubicSpline(self.ss, self.yy)  # generated path_y
        return final_x, final_y
        # 返回两个插值结构体

    def update_current_state(self, new_state_):  # theta can update automatically by current x,y
        self.current_state[0] = new_state_[0]  # x
        self.current_state[1] = new_state_[1]  # y
        self.current_state[2] = new_state_[2]  # heading
        self.current_state[3] = new_state_[3]  # v
        # self.current_state[4] = new_state_[4]  # arc


    def find_closest_point(self, x, y, s0):  # s0是进行迭代求解的初始值，利用scipy库中的优化函数来求得最小值
        # time_start = time.time()

        # 输入的x,y是当前汽车的坐标
        # 返回的是当前汽车对应的是弧长的哪一段以及当前坐标对应的弧长
        def calc_distance(_s, *args):  # 需要进行优化的函数
            _x, _y = self.final_ref_path_x(_s), self.final_ref_path_y(_s)
            return (_x - args[0]) ** 2 + (_y - args[1]) ** 2

        def calc_distance_jacobian(_s, *args):  # 优化函数的雅克比
            _x, _y = self.final_ref_path_x(_s), self.final_ref_path_y(_s)
            _dx, _dy = self.final_ref_path_x.derivative(1)(_s), self.final_ref_path_y.derivative(1)(_s)  # 括号里面的1表示一阶导数
            return 2 * _dx * (_x - args[0]) + 2 * _dy * (_y - args[1])

        minimum = optimize.fmin_cg(calc_distance, s0, calc_distance_jacobian, args=(x, y), full_output=True, disp=False)
        s_best = minimum[0][0]  # 输入优化结果中最好的s值
        i = 0
        # time_end = time.time()
        # print('time cost', time_end - time_start, 's')

        while self.ss[i] <= (s_best+0.2):  # 找到第一个大于s_best的点
            # 进入到里面之后就是第一个比它大的i值
            # if i != (self.ss.size):
            #     i += 1  # 每一次加1
            if (i+3) == self.ss.size-2:  # 由于是以0为索引开始的时候，所以
                # print("GOAL REACHED")
                return s_best, i-1  # 即使是18，也要继续往后传入参数
            else:
                i += 1
        return s_best, i-1

    def run_solver(self, packed_data, going_flag, stop_go):
        # print("MPC Using vref= " + str(self.reference_velocity))
        if self.waypoints_size > 0:
            # first we need get the initial state
            self.solver_xinit[0] = self.current_state[0]  # x
            self.solver_xinit[1] = self.current_state[1]  # y
            self.solver_xinit[2] = self.current_state[2]  # delta
            self.solver_xinit[3] = self.current_state[3]  # v
            self.solver_xinit[4] = 0.0  # arc length (randomly choose a value), next we will update 对应的参考线上的弧长

            # next we need to get the real arc length by initial state of ego vehicle
            s_guess = self.solver_x0[self.solver_total_V + 8]  # Suppose it is the predicted value at the previous horizon
            # print('-----', self.current_state[0], self.current_state[1], s_guess)
            smin, self.traj_i = self.find_closest_point(self.current_state[0], self.current_state[1], s_guess)
            self.solver_xinit[4] = smin  # update achieve
            # self.current_state[4] = smin

            # # 移动障碍物车的位置
            # if self.flag >= 300:
            #     self.ObstOne_x -= 0.3  # 移动障碍物1的x位置

            # if 180 <= self.flag :  # 移动障碍物2的y位置
            #     self.ObstTwo_x += 0.3

            # 后面就开始增加运行时的参数了
            for N_iter in range(self.solver_N):
                k = N_iter * self.solver_NPAR  # i*60
                # the circle to represent ego_vehicle
                self.solver_all_parameters[k + 0] = 1.30  # disc_position1
                self.solver_all_parameters[k + 1] = 1.30  # disc_radius1
                self.solver_all_parameters[k + 2] = -1.30  # disc_position2
                self.solver_all_parameters[k + 3] = 1.30  # disc_radius2
                self.solver_all_parameters[k + 4] = 0.0  # disc_position3
                self.solver_all_parameters[k + 5] = 1.30  # disc_radius3

                # weight
                self.solver_all_parameters[k + 6] = self.error_contour  # contour weight
                self.solver_all_parameters[k + 7] = self.error_lag  # lag weight
                self.solver_all_parameters[k + 8] = self.velocity_weight  # refence

                # 控制量权重
                self.solver_all_parameters[k + 9] = self.acceleration_weight
                self.solver_all_parameters[k + 10] = self.delta_weight
                self.solver_all_parameters[k + 11] = self.Vs_weight  
                self.solver_all_parameters[k + 12] = self.slack_weight

                # reference velocity
                self.solver_all_parameters[k + 13] = self.reference_velocity

                # spline coefficients
                self.solver_all_parameters[k + 14] = self.final_ref_path_x.c[0, self.traj_i]  # a_1
                self.solver_all_parameters[k + 15] = self.final_ref_path_x.c[1, self.traj_i]  # b_1
                self.solver_all_parameters[k + 16] = self.final_ref_path_x.c[2, self.traj_i]  # c_1
                self.solver_all_parameters[k + 17] = self.final_ref_path_x.c[3, self.traj_i]  # d_1

                self.solver_all_parameters[k + 18] = self.final_ref_path_y.c[0, self.traj_i]  # a_1
                self.solver_all_parameters[k + 19] = self.final_ref_path_y.c[1, self.traj_i]  # b_1
                self.solver_all_parameters[k + 20] = self.final_ref_path_y.c[2, self.traj_i]  # c_1
                self.solver_all_parameters[k + 21] = self.final_ref_path_y.c[3, self.traj_i]  # d_1

                self.solver_all_parameters[k + 22] = self.final_ref_path_x.c[0, self.traj_i + 1]  # a_2
                self.solver_all_parameters[k + 23] = self.final_ref_path_x.c[1, self.traj_i + 1]  # b_2
                self.solver_all_parameters[k + 24] = self.final_ref_path_x.c[2, self.traj_i + 1]  # c_2
                self.solver_all_parameters[k + 25] = self.final_ref_path_x.c[3, self.traj_i + 1]  # d_2

                self.solver_all_parameters[k + 26] = self.final_ref_path_y.c[0, self.traj_i + 1]  # a_2
                self.solver_all_parameters[k + 27] = self.final_ref_path_y.c[1, self.traj_i + 1]  # b_2
                self.solver_all_parameters[k + 28] = self.final_ref_path_y.c[2, self.traj_i + 1]  # c_2
                self.solver_all_parameters[k + 29] = self.final_ref_path_y.c[3, self.traj_i + 1]  # d_2

                # 下面是另外增加了几条曲线,还可以继续添加
                self.solver_all_parameters[k + 30] = self.final_ref_path_x.c[0, self.traj_i + 2]  # a_3
                self.solver_all_parameters[k + 31] = self.final_ref_path_x.c[1, self.traj_i + 2]  # b_3
                self.solver_all_parameters[k + 32] = self.final_ref_path_x.c[2, self.traj_i + 2]  # c_3
                self.solver_all_parameters[k + 33] = self.final_ref_path_x.c[3, self.traj_i + 2]  # d_3

                self.solver_all_parameters[k + 34] = self.final_ref_path_y.c[0, self.traj_i + 2]  # a_3
                self.solver_all_parameters[k + 35] = self.final_ref_path_y.c[1, self.traj_i + 2]  # b_3
                self.solver_all_parameters[k + 36] = self.final_ref_path_y.c[2, self.traj_i + 2]  # c_3
                self.solver_all_parameters[k + 37] = self.final_ref_path_y.c[3, self.traj_i + 2]  # d_3

                self.solver_all_parameters[k + 38] = self.final_ref_path_x.c[0, self.traj_i + 3]  # a_4
                self.solver_all_parameters[k + 39] = self.final_ref_path_x.c[1, self.traj_i + 3]  # b_4
                self.solver_all_parameters[k + 40] = self.final_ref_path_x.c[2, self.traj_i + 3]  # c_4
                self.solver_all_parameters[k + 41] = self.final_ref_path_x.c[3, self.traj_i + 3]  # d_4

                self.solver_all_parameters[k + 42] = self.final_ref_path_y.c[0, self.traj_i + 3]  # a_4
                self.solver_all_parameters[k + 43] = self.final_ref_path_y.c[1, self.traj_i + 3]  # b_4
                self.solver_all_parameters[k + 44] = self.final_ref_path_y.c[2, self.traj_i + 3]  # c_4
                self.solver_all_parameters[k + 45] = self.final_ref_path_y.c[3, self.traj_i + 3]  # d_4

                self.solver_all_parameters[k + 46] = self.final_ref_path_x.c[0, self.traj_i + 4]  # a_5
                self.solver_all_parameters[k + 47] = self.final_ref_path_x.c[1, self.traj_i + 4]  # b_5
                self.solver_all_parameters[k + 48] = self.final_ref_path_x.c[2, self.traj_i + 4]  # c_5
                self.solver_all_parameters[k + 49] = self.final_ref_path_x.c[3, self.traj_i + 4]  # d_5

                self.solver_all_parameters[k + 50] = self.final_ref_path_y.c[0, self.traj_i + 4]  # a_5
                self.solver_all_parameters[k + 51] = self.final_ref_path_y.c[1, self.traj_i + 4]  # b_5
                self.solver_all_parameters[k + 52] = self.final_ref_path_y.c[2, self.traj_i + 4]  # c_5
                self.solver_all_parameters[k + 53] = self.final_ref_path_y.c[3, self.traj_i + 4]  # d_5

                # The end point of the curve
                self.solver_all_parameters[k + 54] = self.ss[self.traj_i]  # s1 the starting point of first  curve
                self.solver_all_parameters[k + 55] = self.ss[self.traj_i + 1]  # s2 the starting point of second curve
                self.solver_all_parameters[k + 56] = self.ss[self.traj_i + 2]  # s3
                self.solver_all_parameters[k + 57] = self.ss[self.traj_i + 3]  # s4
                self.solver_all_parameters[k + 58] = self.ss[self.traj_i + 4]  # s5

                # ----------------------------------Obstalce Information----------------------------------#
                # packed_data.shape=[12,30,7]
                self.solver_all_parameters[k + 59] = packed_data[0, N_iter, 0]  # ObstOne_x
                self.solver_all_parameters[k + 60] = packed_data[0, N_iter, 1]  # ObstOne_y
                self.solver_all_parameters[k + 61] = packed_data[0, N_iter, 2]  # deg2rad(90)  # ObstOne_heading
                self.solver_all_parameters[k + 62] = packed_data[0, N_iter, 3]
                self.solver_all_parameters[k + 63] = packed_data[0, N_iter, 4]  # + packed_data[0, N_iter, 6]  # ObstOne_minor_axis

                self.solver_all_parameters[k + 64] = packed_data[1, N_iter, 0]  # ObstTwo_x
                self.solver_all_parameters[k + 65] = packed_data[1, N_iter, 1]  # ObstTwo_y
                self.solver_all_parameters[k + 66] = packed_data[1, N_iter, 2]  # ObstTwo_heading
                self.solver_all_parameters[k + 67] = packed_data[1, N_iter, 3]
                self.solver_all_parameters[k + 68] = packed_data[1, N_iter, 4]  # + packed_data[1, N_iter, 6]

                # Obstacle 3-4
                self.solver_all_parameters[k + 69] = packed_data[2, N_iter, 0]  # ObstThree_x
                self.solver_all_parameters[k + 70] = packed_data[2, N_iter, 1]  # ObstThree_y
                self.solver_all_parameters[k + 71] = packed_data[2, N_iter, 2]  # ObstThree_heading
                self.solver_all_parameters[k + 72] = packed_data[2, N_iter, 3]
                self.solver_all_parameters[k + 73] = packed_data[2, N_iter, 4]  # + packed_data[2, N_iter, 6]

                self.solver_all_parameters[k + 74] = packed_data[3, N_iter, 0]  # ObstFour_x
                self.solver_all_parameters[k + 75] = packed_data[3, N_iter, 1]  # ObstFour_y
                self.solver_all_parameters[k + 76] = packed_data[3, N_iter, 2]  # ObstFour_heading
                self.solver_all_parameters[k + 77] = packed_data[3, N_iter, 3]
                self.solver_all_parameters[k + 78] = packed_data[3, N_iter, 4]  # + packed_data[3, N_iter, 6]

                # Obstacle 5-6
                self.solver_all_parameters[k + 79] = packed_data[4, N_iter, 0]  # ObstFive_x
                self.solver_all_parameters[k + 80] = packed_data[4, N_iter, 1]  # ObstFive_y
                self.solver_all_parameters[k + 81] = packed_data[4, N_iter, 2]  # ObstFive_heading
                self.solver_all_parameters[k + 82] = packed_data[4, N_iter, 3]
                self.solver_all_parameters[k + 83] = packed_data[4, N_iter, 4]  # + packed_data[4, N_iter, 6]  # ObstFive_minor_axis

                self.solver_all_parameters[k + 84] = packed_data[5, N_iter, 0]  # ObstSix_x
                self.solver_all_parameters[k + 85] = packed_data[5, N_iter, 1]  # ObstSix_y
                self.solver_all_parameters[k + 86] = packed_data[5, N_iter, 2]  # ObstSix_heading
                self.solver_all_parameters[k + 87] = packed_data[5, N_iter, 3]
                self.solver_all_parameters[k + 88] = packed_data[5, N_iter, 4]  # + packed_data[5, N_iter, 6]

                # Obstacle 7-8
                self.solver_all_parameters[k + 89] = packed_data[6, N_iter, 0]  # ObstSeven_x
                self.solver_all_parameters[k + 90] = packed_data[6, N_iter, 1]  # ObstSeven_y
                self.solver_all_parameters[k + 91] = packed_data[6, N_iter, 2]  # ObstSeven_heading
                self.solver_all_parameters[k + 92] = packed_data[6, N_iter, 3] # ObstSeven_major_axis
                self.solver_all_parameters[k + 93] = packed_data[6, N_iter, 4] # + packed_data[6, N_iter, 6]  # ObstSeven_minor_axis

                self.solver_all_parameters[k + 94] = packed_data[7, N_iter, 0]  # ObstEight_x
                self.solver_all_parameters[k + 95] = packed_data[7, N_iter, 1]  # ObstEight_y
                self.solver_all_parameters[k + 96] = packed_data[7, N_iter, 2]  # ObstEight_heading
                self.solver_all_parameters[k + 97] = packed_data[7, N_iter, 3] # ObstEight_major_axis
                self.solver_all_parameters[k + 98] = packed_data[7, N_iter, 4]  # + packed_data[7, N_iter, 6]  # ObstEight_minor_axis

                # Obstacle 9-10
                self.solver_all_parameters[k + 99] = packed_data[8, N_iter, 0]  # ObstNine_x
                self.solver_all_parameters[k + 100] = packed_data[8, N_iter, 1]  # ObstNine_y
                self.solver_all_parameters[k + 101] = packed_data[8, N_iter, 2]  # ObstNine_heading
                self.solver_all_parameters[k + 102] = packed_data[8, N_iter, 3] # ObstNine_major_axis
                self.solver_all_parameters[k + 103] = packed_data[8, N_iter, 4]  # + packed_data[8, N_iter, 6]  # ObstNine_minor_axis

                self.solver_all_parameters[k + 104] = packed_data[9, N_iter, 0]  # ObstTen_x
                self.solver_all_parameters[k + 105] = packed_data[9, N_iter, 1]  # ObstTen_y
                self.solver_all_parameters[k + 106] = packed_data[9, N_iter, 2]  # ObstTen_heading
                self.solver_all_parameters[k + 107] = packed_data[9, N_iter, 3]  # ObstTen_major_axis
                self.solver_all_parameters[k + 108] = packed_data[9, N_iter, 4]  # + packed_data[9, N_iter, 6]  # ObstTen_minor_axis

                # Obstacle 11-12
                self.solver_all_parameters[k + 109] = packed_data[10, N_iter, 0]  # ObstEleven_x
                self.solver_all_parameters[k + 110] = packed_data[10, N_iter, 1]  # ObstEleven_y
                self.solver_all_parameters[k + 111] = packed_data[10, N_iter, 2]  # ObstEleven_heading
                self.solver_all_parameters[k + 112] = packed_data[10, N_iter, 3] # ObstEleven_major_axis
                self.solver_all_parameters[k + 113] = packed_data[10, N_iter, 4]  # + packed_data[10, N_iter, 6]  # ObstEleven_minor_axis

                self.solver_all_parameters[k + 114] = packed_data[11, N_iter, 0]  # ObstTwelve_x
                self.solver_all_parameters[k + 115] = packed_data[11, N_iter, 1]  # ObstTwelve_y
                self.solver_all_parameters[k + 116] = packed_data[11, N_iter, 2]  # ObstTwelve_heading
                self.solver_all_parameters[k + 117] = packed_data[11, N_iter, 3] # ObstTwelve_major_axis
                self.solver_all_parameters[k + 118] = packed_data[11, N_iter, 4]  # + packed_data[11, N_iter, 6]  # ObstTwelve_minor_axis

            # the problem established completely
            problem = {"x0": self.solver_x0, "xinit": self.solver_xinit, "all_parameters": self.solver_all_parameters}
            output, exitflag, info = solver.solve(problem)
            self.flag += 1
            # print(self.flag)
            if self.current_state[3] <=0.1:
                    self.stop_step += 1
            else:
                self.stop_step =0

            if exitflag != 1:
                # print("The exitflag is", exitflag)
                self.reference_velocity = 0
                if self.stop_step >= 5 and going_flag == True or stop_go == True:
                    self.reference_velocity = 10
            else:
                self.reference_velocity = 10
            # self.reference_velocity = 10

            # next we will process data
            # every line of all_value is [(a,delta,v_s,s_v) (x,y,phi,v,theta)]
            all_value = np.zeros([self.solver_N, self.solver_total_V])  # 20x9, use for save all results
            for i in range(self.solver_N):
                a = "x{:02d}".format(i + 1)
                all_value[i, :] = output[a]


            # take this solved result as the initial guess of next solving
            for i in range(self.solver_N):
                k = i*self.solver_total_V  # i*9
                for j in range(self.solver_total_V):
                    self.solver_x0[k+j] = all_value[i, j]

            return all_value, exitflag, info, self.final_ref_path_x, self.final_ref_path_y, self.ss, self.current_state, self.traj_i # 20*9
        else:
            print("NO WAYPOINTS PROVIDED")
