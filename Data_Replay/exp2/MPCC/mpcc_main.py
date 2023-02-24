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
import matplotlib
from scipy import optimize
from matplotlib.patches import Ellipse, Circle, Rectangle
import matplotlib.gridspec as gridspec
import time
from visualizations.map_vis_xy_SinD import draw_map_without_lanelet_xy
import gif
from celluloid import Camera
from matplotlib.animation import FFMpegWriter
solver = forcespro.nlp.Solver.from_directory("/home/kai/Vscode/Engineering/Decision_making_exp1/MPCC/FORCESNLPsolver")  # load solver
lanelet_map_file = '/home/kai/Vscode/Engineering/SinD_Dataset/doc/mapfile-Tianjin.osm'

def remove_item(n):
    return n != 0

def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center

def polygon_xy(ms, length, width):
    
    lowleft = (ms[0] - length / 2., ms[1] - width / 2.)
    lowright = (ms[0]+ length / 2., ms[1] - width / 2.)
    upright = (ms[0] + length / 2., ms[1] + width / 2.)
    upleft = (ms[0] - length / 2., ms[1] + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms[0], ms[1]]), yaw=ms[2])


class MPCC:
    def __init__(self, current_state_):

        my_track = pd.read_csv("/home/kai/Vscode/Engineering/Decision_making_exp2/MPCC/track_8_9_4_Veh489.csv")
        # x_ref and y_ref is all waypoints
        self.x_ref = my_track.loc[0].values  # convet to the numpy ndarray
        
        self.y_ref = my_track.loc[1].values
        
        self.waypoints_size = self.x_ref.size  # total number of waypoints
        
        self.n_pts = 20  # we need 20 points to represent the whole path

        
        # the value of 20 points
        self.ss = np.zeros([self.n_pts])
        self.xx = np.zeros([self.n_pts])
        self.yy = np.zeros([self.n_pts])
        # to represent path
        self.final_ref_path_x, self.final_ref_path_y = self.ref_path()
        
        self.current_state = current_state_  # (x,y,phi,v)

        # the MPCC parameters
        self.solver_N = 30
        self.solver_NU = 4  # s.t. (a,delta,v_virtual,slack_value)
        self.solver_NX = 5  # s.t. (x,y,phi,v,arc_length)
        self.solver_total_V = 9
        self.solver_NPAR = 117  # all parameters received by solver
        x0i = np.array([0., 0., 0., 0., -23.33, 14.63, 0.031, 5, 0], dtype="double")

        self.solver_x0 = np.transpose(np.tile(x0i, (1, self.solver_N)))
        # the input value to solver
        # self.solver_x0 = np.zeros(self.solver_N*self.solver_total_V)  # the x0 to solve problem
        self.solver_xinit = np.zeros(self.solver_NX, dtype="double")  # the initial state of MPCC
        self.solver_all_parameters = np.zeros(self.solver_NPAR * self.solver_N, dtype="double")  # the runtime parameters

        # weights
        self.velocity_weight = 4
        self.error_contour = 130
        self.error_lag = 130

        self.acceleration_weight = 6
        self.delta_weight = 100
        self.slack_weight = 2800  # slack variables
        self.Vs_weight = 0.1  # v_virtual(s.t.,velocity along the arc)

        # the desired velocity
        self.reference_velocity = 5

        self.stop_step =0
 
        self.traj_i = 0
        # flag
        self.flag = 0  # running times of run_solver


    def ref_path(self):
        x_coordinate = self.x_ref
        y_coordinate = self.y_ref
        waypoints_size = self.waypoints_size  # int
        t = np.arange(0, waypoints_size)  # [0,1,2,...,99], total number is 100
        cs_x = CubicSpline(t, x_coordinate)  # class
        cs_y = CubicSpline(t, y_coordinate)  # class
        de_x = cs_x.derivative()  # calculate the derivative
        de_y = cs_y.derivative()
        u_min = cs_x.x[:-1] 
        u_max = cs_x.x[1:]
        all_si = np.zeros([u_min.size])

        for i in range(u_min.size):
            f = lambda c: np.sqrt(de_x(c) ** 2 + de_y(c) ** 2)  # the function about the const value c
            si = scipy.integrate.quad(f, u_min[i], u_max[i])  # quad function returns two values, integration values and error
            all_si[i] = si[0]
        # the consum arc length
        s = np.append([0], np.cumsum(all_si))  # the size is 100
        total_length = s[-1]  # 111.0164
        # then we build our spline
        s_x = CubicSpline(s, x_coordinate)
        s_y = CubicSpline(s, y_coordinate)
        dist_spline_pts = total_length / (self.n_pts - 1)

        for i in range(self.n_pts):  # (0,1,...19) total 20
            self.ss[i] = dist_spline_pts * i
            self.xx[i] = s_x(self.ss[i])
            self.yy[i] = s_y(self.ss[i])
        final_x = CubicSpline(self.ss, self.xx)  # generated path_x
        final_y = CubicSpline(self.ss, self.yy)  # generated path_y
        return final_x, final_y

    def update_current_state(self, new_state_):  # theta can update automatically by current x,y
        self.current_state[0] = new_state_[0]  # x
        self.current_state[1] = new_state_[1]  # y
        self.current_state[2] = new_state_[2]  # heading
        self.current_state[3] = new_state_[3]  # v
        # self.current_state[4] = new_state_[4]  # arc


    def find_closest_point(self, x, y, s0): 
        
        def calc_distance(_s, *args): 
            _x, _y = self.final_ref_path_x(_s), self.final_ref_path_y(_s)
            return (_x - args[0]) ** 2 + (_y - args[1]) ** 2

        def calc_distance_jacobian(_s, *args):
            _x, _y = self.final_ref_path_x(_s), self.final_ref_path_y(_s)
            _dx, _dy = self.final_ref_path_x.derivative(1)(_s), self.final_ref_path_y.derivative(1)(_s)  # 括号里面的1表示一阶导数
            return 2 * _dx * (_x - args[0]) + 2 * _dy * (_y - args[1])

        minimum = optimize.fmin_cg(calc_distance, s0, calc_distance_jacobian, args=(x, y), full_output=True, disp=False)
        s_best = minimum[0][0] 
        i = 0

        while self.ss[i] <= (s_best+0.2):

            if (i+3) == self.ss.size-2: 
                # print("GOAL REACHED")
                return s_best, i-1 
            else:
                i += 1
        return s_best, i-1

    def run_solver(self, packed_data, going_flag, stop_go):
        
        if self.waypoints_size > 0:
            # first we need get the initial state
            self.solver_xinit[0] = self.current_state[0]  # x
            self.solver_xinit[1] = self.current_state[1]  # y
            self.solver_xinit[2] = self.current_state[2]  # delta
            self.solver_xinit[3] = self.current_state[3]  # v
            self.solver_xinit[4] = 0.0  # arc length (randomly choose a value), next we will update 对应的参考线上的弧长

            s_guess = self.solver_x0[self.solver_total_V + 8]  # Suppose it is the predicted value at the previous horizon
            
            smin, self.traj_i = self.find_closest_point(self.current_state[0], self.current_state[1], s_guess)
            self.solver_xinit[4] = smin  # update achieve
            # self.current_state[4] = smin

            for N_iter in range(self.solver_N):
                k = N_iter * self.solver_NPAR  # i*60
                # the circle to represent ego_vehicle
                self.solver_all_parameters[k + 0] = 0.6  # disc_position1
                self.solver_all_parameters[k + 1] = 1.2  # disc_radius1
                self.solver_all_parameters[k + 2] = -0.6  # disc_position2
                self.solver_all_parameters[k + 3] = 1.2  # disc_radius2
                # self.solver_all_parameters[k + 0] = 0.5  # disc_position1
                # self.solver_all_parameters[k + 1] = 0.8  # disc_radius1
                # self.solver_all_parameters[k + 2] = -0.5  # disc_position2
                # self.solver_all_parameters[k + 3] = 0.8  # disc_radius2

                # weight
                self.solver_all_parameters[k + 4] = self.error_contour  # contour weight
                self.solver_all_parameters[k + 5] = self.error_lag  # lag weight
                self.solver_all_parameters[k + 6] = self.velocity_weight  # refence

                self.solver_all_parameters[k + 7] = self.acceleration_weight
                self.solver_all_parameters[k + 8] = self.delta_weight
                self.solver_all_parameters[k + 9] = self.Vs_weight
                self.solver_all_parameters[k + 10] = self.slack_weight

                # reference velocity
                self.solver_all_parameters[k + 11] = self.reference_velocity

                # spline coefficients
                self.solver_all_parameters[k + 12] = self.final_ref_path_x.c[0, self.traj_i]  # a_1
                self.solver_all_parameters[k + 13] = self.final_ref_path_x.c[1, self.traj_i]  # b_1
                self.solver_all_parameters[k + 14] = self.final_ref_path_x.c[2, self.traj_i]  # c_1
                self.solver_all_parameters[k + 15] = self.final_ref_path_x.c[3, self.traj_i]  # d_1

                self.solver_all_parameters[k + 16] = self.final_ref_path_y.c[0, self.traj_i]  # a_1
                self.solver_all_parameters[k + 17] = self.final_ref_path_y.c[1, self.traj_i]  # b_1
                self.solver_all_parameters[k + 18] = self.final_ref_path_y.c[2, self.traj_i]  # c_1
                self.solver_all_parameters[k + 19] = self.final_ref_path_y.c[3, self.traj_i]  # d_1

                self.solver_all_parameters[k + 20] = self.final_ref_path_x.c[0, self.traj_i + 1]  # a_2
                self.solver_all_parameters[k + 21] = self.final_ref_path_x.c[1, self.traj_i + 1]  # b_2
                self.solver_all_parameters[k + 22] = self.final_ref_path_x.c[2, self.traj_i + 1]  # c_2
                self.solver_all_parameters[k + 23] = self.final_ref_path_x.c[3, self.traj_i + 1]  # d_2

                self.solver_all_parameters[k + 24] = self.final_ref_path_y.c[0, self.traj_i + 1]  # a_2
                self.solver_all_parameters[k + 25] = self.final_ref_path_y.c[1, self.traj_i + 1]  # b_2
                self.solver_all_parameters[k + 26] = self.final_ref_path_y.c[2, self.traj_i + 1]  # c_2
                self.solver_all_parameters[k + 27] = self.final_ref_path_y.c[3, self.traj_i + 1]  # d_2

                self.solver_all_parameters[k + 28] = self.final_ref_path_x.c[0, self.traj_i + 2]  # a_3
                self.solver_all_parameters[k + 29] = self.final_ref_path_x.c[1, self.traj_i + 2]  # b_3
                self.solver_all_parameters[k + 30] = self.final_ref_path_x.c[2, self.traj_i + 2]  # c_3
                self.solver_all_parameters[k + 31] = self.final_ref_path_x.c[3, self.traj_i + 2]  # d_3

                self.solver_all_parameters[k + 32] = self.final_ref_path_y.c[0, self.traj_i + 2]  # a_3
                self.solver_all_parameters[k + 33] = self.final_ref_path_y.c[1, self.traj_i + 2]  # b_3
                self.solver_all_parameters[k + 34] = self.final_ref_path_y.c[2, self.traj_i + 2]  # c_3
                self.solver_all_parameters[k + 35] = self.final_ref_path_y.c[3, self.traj_i + 2]  # d_3

                self.solver_all_parameters[k + 36] = self.final_ref_path_x.c[0, self.traj_i + 3]  # a_4
                self.solver_all_parameters[k + 37] = self.final_ref_path_x.c[1, self.traj_i + 3]  # b_4
                self.solver_all_parameters[k + 38] = self.final_ref_path_x.c[2, self.traj_i + 3]  # c_4
                self.solver_all_parameters[k + 39] = self.final_ref_path_x.c[3, self.traj_i + 3]  # d_4

                self.solver_all_parameters[k + 40] = self.final_ref_path_y.c[0, self.traj_i + 3]  # a_4
                self.solver_all_parameters[k + 41] = self.final_ref_path_y.c[1, self.traj_i + 3]  # b_4
                self.solver_all_parameters[k + 42] = self.final_ref_path_y.c[2, self.traj_i + 3]  # c_4
                self.solver_all_parameters[k + 43] = self.final_ref_path_y.c[3, self.traj_i + 3]  # d_4

                self.solver_all_parameters[k + 44] = self.final_ref_path_x.c[0, self.traj_i + 4]  # a_5
                self.solver_all_parameters[k + 45] = self.final_ref_path_x.c[1, self.traj_i + 4]  # b_5
                self.solver_all_parameters[k + 46] = self.final_ref_path_x.c[2, self.traj_i + 4]  # c_5
                self.solver_all_parameters[k + 47] = self.final_ref_path_x.c[3, self.traj_i + 4]  # d_5

                self.solver_all_parameters[k + 48] = self.final_ref_path_y.c[0, self.traj_i + 4]  # a_5
                self.solver_all_parameters[k + 49] = self.final_ref_path_y.c[1, self.traj_i + 4]  # b_5
                self.solver_all_parameters[k + 50] = self.final_ref_path_y.c[2, self.traj_i + 4]  # c_5
                self.solver_all_parameters[k + 51] = self.final_ref_path_y.c[3, self.traj_i + 4]  # d_5

                # The end point of the curve
                self.solver_all_parameters[k + 52] = self.ss[self.traj_i]      # s1 the starting point of first  curve
                self.solver_all_parameters[k + 53] = self.ss[self.traj_i + 1]  # s2 the starting point of second curve
                self.solver_all_parameters[k + 54] = self.ss[self.traj_i + 2]  # s3
                self.solver_all_parameters[k + 55] = self.ss[self.traj_i + 3]  # s4
                self.solver_all_parameters[k + 56] = self.ss[self.traj_i + 4]  # s5

                #----------------------------------Obstalce Information----------------------------------#

                self.solver_all_parameters[k + 57] = packed_data[0, N_iter, 0] # ObstOne_x
                self.solver_all_parameters[k + 58] = packed_data[0, N_iter, 1]  # ObstOne_y
                self.solver_all_parameters[k + 59] = packed_data[0, N_iter, 2]  # deg2rad(90)  # ObstOne_heading
                self.solver_all_parameters[k + 60] = packed_data[0, N_iter, 3]
                self.solver_all_parameters[k + 61] = packed_data[0, N_iter, 4]  # ObstOne_minor_axis

                self.solver_all_parameters[k + 62] = packed_data[1, N_iter, 0]  # ObstTwo_x
                self.solver_all_parameters[k + 63] = packed_data[1, N_iter, 1]  # ObstTwo_y
                self.solver_all_parameters[k + 64] = packed_data[1, N_iter, 2]  # ObstTwo_heading
                self.solver_all_parameters[k + 65] = packed_data[1, N_iter, 3]
                self.solver_all_parameters[k + 66] = packed_data[1, N_iter, 4]

                # Obstacle 3-4
                self.solver_all_parameters[k + 67] = packed_data[2, N_iter, 0]  # ObstThree_x
                self.solver_all_parameters[k + 68] = packed_data[2, N_iter, 1]  # ObstThree_y
                self.solver_all_parameters[k + 69] = packed_data[2, N_iter, 2]  # ObstThree_heading
                self.solver_all_parameters[k + 70] = packed_data[2, N_iter, 3]  # ObstThree_major_axis
                self.solver_all_parameters[k + 71] = packed_data[2, N_iter, 4]  # ObstThree_minor_axis
                
                self.solver_all_parameters[k + 72] = packed_data[3, N_iter, 0]  # ObstFour_x
                self.solver_all_parameters[k + 73] = packed_data[3, N_iter, 1]  # ObstFour_y
                self.solver_all_parameters[k + 74] = packed_data[3, N_iter, 2]  # ObstFour_heading
                self.solver_all_parameters[k + 75] = packed_data[3, N_iter, 3]  # ObstFour_major_axis
                self.solver_all_parameters[k + 76] = packed_data[3, N_iter, 4]  # ObstFour_minor_axis

                # Obstacle 5-6
                self.solver_all_parameters[k + 77] = packed_data[4, N_iter, 0]  # ObstFive_x
                self.solver_all_parameters[k + 78] = packed_data[4, N_iter, 1]  # ObstFive_y
                self.solver_all_parameters[k + 79] = packed_data[4, N_iter, 2]  # ObstFive_heading
                self.solver_all_parameters[k + 80] = packed_data[4, N_iter, 3]  # ObstFive_major_axis
                self.solver_all_parameters[k + 81] = packed_data[4, N_iter, 4]  # ObstFive_minor_axis
                
                self.solver_all_parameters[k + 82] = packed_data[5, N_iter, 0]  # ObstSix_x
                self.solver_all_parameters[k + 83] = packed_data[5, N_iter, 1]  # ObstSix_y
                self.solver_all_parameters[k + 84] = packed_data[5, N_iter, 2]  # ObstSix_heading
                self.solver_all_parameters[k + 85] = packed_data[5, N_iter, 3]  # ObstSix_major_axis
                self.solver_all_parameters[k + 86] = packed_data[5, N_iter, 4]  # ObstSix_minor_axis

                # Obstacle 7-8
                self.solver_all_parameters[k + 87] = packed_data[6, N_iter, 0]  # ObstSeven_x
                self.solver_all_parameters[k + 88] = packed_data[6, N_iter, 1]  # ObstSeven_y
                self.solver_all_parameters[k + 89] = packed_data[6, N_iter, 2]  # ObstSeven_heading
                self.solver_all_parameters[k + 90] = packed_data[6, N_iter, 3]  # ObstSeven_major_axis
                self.solver_all_parameters[k + 91] = packed_data[6, N_iter, 4]  # ObstSeven_minor_axis
                
                self.solver_all_parameters[k + 92] = packed_data[7, N_iter, 0]  # ObstEight_x
                self.solver_all_parameters[k + 93] = packed_data[7, N_iter, 1]  # ObstEight_y
                self.solver_all_parameters[k + 94] = packed_data[7, N_iter, 2]  # ObstEight_heading
                self.solver_all_parameters[k + 95] = packed_data[7, N_iter, 3]  # ObstEight_major_axis
                self.solver_all_parameters[k + 96] = packed_data[7, N_iter, 4]  # ObstEight_minor_axis

                # Obstacle 9-10
                self.solver_all_parameters[k + 97] = packed_data[8, N_iter, 0]  # ObstNine_x
                self.solver_all_parameters[k + 98] = packed_data[8, N_iter, 1]  # ObstNine_y
                self.solver_all_parameters[k + 99] = packed_data[8, N_iter, 2]  # ObstNine_heading
                self.solver_all_parameters[k + 100] = packed_data[8, N_iter, 3]  # ObstNine_major_axis
                self.solver_all_parameters[k + 101] = packed_data[8, N_iter, 4]  # ObstNine_minor_axis
                
                self.solver_all_parameters[k + 102] = packed_data[9, N_iter, 0]  # ObstTen_x
                self.solver_all_parameters[k + 103] = packed_data[9, N_iter, 1]  # ObstTen_y
                self.solver_all_parameters[k + 104] = packed_data[9, N_iter, 2]  # ObstTen_heading
                self.solver_all_parameters[k + 105] = packed_data[9, N_iter, 3]  # ObstTen_major_axis
                self.solver_all_parameters[k + 106] = packed_data[9, N_iter, 4]  # ObstTen_minor_axis

                # Obstacle 11-12
                self.solver_all_parameters[k + 107] = packed_data[10, N_iter, 0]  # ObstEleven_x
                self.solver_all_parameters[k + 108] = packed_data[10, N_iter, 1]   # ObstEleven_y
                self.solver_all_parameters[k + 109] = packed_data[10, N_iter, 2]   # ObstEleven_heading
                self.solver_all_parameters[k + 110] = packed_data[10, N_iter, 3]   # ObstEleven_major_axis
                self.solver_all_parameters[k + 111] = packed_data[10, N_iter, 4]   # ObstEleven_minor_axis
                
                self.solver_all_parameters[k + 112] = packed_data[11, N_iter, 0]  # ObstTwelve_x
                self.solver_all_parameters[k + 113] = packed_data[11, N_iter, 1]  # ObstTwelve_y
                self.solver_all_parameters[k + 114] = packed_data[11, N_iter, 2]  # ObstTwelve_heading
                self.solver_all_parameters[k + 115] = packed_data[11, N_iter, 3]  # ObstTwelve_major_axis
                self.solver_all_parameters[k + 116] = packed_data[11, N_iter, 4]  # ObstTwelve_minor_axis

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
                print("The exitflag is", exitflag)
                self.reference_velocity = 0
                if self.stop_step >= 5 and going_flag == True and stop_go==True:
                    self.reference_velocity = 6
            else:
                self.reference_velocity =6
            # self.reference_velocity =6

            sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n".format(info.it, info.solvetime))

            all_value = np.zeros([self.solver_N, self.solver_total_V])  # 20x9, use for save all results
            for i in range(self.solver_N):
                a = "x{:02d}".format(i + 1)
                all_value[i, :] = output[a]


            # take this solved result as the initial guess of next solving
            for i in range(self.solver_N):
                k = i*self.solver_total_V  # i*9
                for j in range(self.solver_total_V):
                    self.solver_x0[k+j] = all_value[i, j]

            return all_value, exitflag, info, self.final_ref_path_x, self.final_ref_path_y, self.ss, self.current_state, self.traj_i
        else:
            print("NO WAYPOINTS PROVIDED")

