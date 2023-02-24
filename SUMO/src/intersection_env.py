from cmath import inf
import os
import sys
import numpy as np
import copy
import warnings
from pandas import array
import torch
from torch_geometric.data import Data
import math
from PIL import Image, ImageOps
from vehicle_model import vehicle_model
from intersect_detection import rotate_box
from intersect_detection import intersect_detection
import casadi
import scipy
import scipy.integrate
from scipy import optimize
import forcespro
import matplotlib.pyplot as plt
from GNN.visualizations import map_vis_xy
import GNN.utils


# from sqlalchemy import false
warnings.simplefilter('always', UserWarning)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import traci
from road import Road

# Sumo subscription constants
POSITION = 66
LONG_SPEED = 64
LAT_SPEED = 50
LONG_ACC = 114
ANGLE = 67


class IntersectionEnv(object):
    """
    This class creates a gym-like intersection driving environment.

    The ego vehicle starts in the south and aims to cross the intersection to the north. Surrounding traffic
    is initialized at random in the west and east, with intentions to cross the intersection, turn to the north,
    or the south. The surrounding traffic follows the SUMO driver model, except for that the presence of the
    ego vehicle is ignored. See the paper for more details.

    The parameters of the environment are defined in parameters_intersection.py.
    The environment is built in a gym-like structure, with the methods 'reset' and 'step'

    Args:
        sim_params: Parameters that describe the simulation setup, the action space, and the reward function
        road_params: Parameters that describe the road geometry, rules, and properties of different vehicles
        gui_params (bool): GUI options.
        start_time (str): Optional label
    """
    def __init__(self, sim_params, road_params, gui_params=None, start_time=''):
        
        self.goal = 'left-turn' # 'go-straight', 'right-turn'
        self.step_ = 0

        # Parameters road
        self.intersection_pos = road_params['intersection_position']
        self.nb_lanes = road_params['nb_lanes']
        self.lane_width = road_params['lane_width']
        self.ego_length = road_params['vehicles'][0]['length']
        self.ego_width = road_params['vehicles'][0]['width']
        self.car_length = road_params['vehicles'][1]['length']
        self.car_width = road_params['vehicles'][1]['width']
        self.moped_length = 2.1
        self.moped_width = 0.8
        self.stop_line = road_params['stop_line']

        # Parameters scenario
        self.max_steps = sim_params['max_steps']*10
        self.start_pos = sim_params['ego_start_position']
        self.end_pos = sim_params['ego_end_position']
        self.start_route = sim_params['ego_start_route']
        self.start_lane = sim_params['ego_start_lane']
        self.init_steps = sim_params['init_steps']
        self.adding_prob = sim_params['adding_prob']
        self.ped_adding_prob = sim_params['ped_adding_prob']
        self.mop_adding_prob = sim_params['mop_adding_prob']
        self.max_nb_vehicles = sim_params['nb_vehicles']
        self.max_nb_pedestrians = sim_params['nb_pedestrians']
        self.max_nb_moped = sim_params['nb_moped']
        self.idm_params = sim_params['idm_params']
        self.max_speed = road_params['speed_range'][1]
        self.min_speed = road_params['speed_range'][0]
        self.max_ego_speed = road_params['vehicles'][0]['maxSpeed']
        self.safety_check = sim_params['safety_check']

        # Parameters sensing
        self.nb_ego_states = 8  # position-x, position-y, velocity, heading, done, light_state, light_dis, switch_time_remain
        self.nb_states_per_vehicle = 4
        self.nb_states_per_moped = 4
        self.nb_states_per_pedestrian = 4
        self.sensor_range = sim_params['sensor_range']
        self.occlusion_dist = sim_params['occlusion_dist']
        self.sensor_nb_vehicles = sim_params['sensor_nb_vehicles']
        self.sensor_nb_pedestrians = sim_params['sensor_nb_pedestrians']
        self.sensor_nb_moped = sim_params['sensor_nb_moped']
        self.sensor_noise = sim_params['sensor_noise']
        self.sensor_max_speed_scale = self.max_speed
        self.sensor_mop_max_speed_scale = 7.5
        self.sensor_ped_max_speed_scale = 3
        # Parameters reward
        self.goal_reward = sim_params['goal_reward']
        self.collision_penalty = sim_params['collision_penalty']
        self.near_collision_penalty = sim_params['near_collision_penalty']
        self.near_collision_margin = sim_params['near_collision_margin']
        self.pedestrian_collision_margin = 1
        self.mop_collision_margin = 2

        # GUI parameters
        self.use_gui = gui_params['use_gui'] if gui_params else False
        self.print_gui_info = gui_params['print_gui_info'] if gui_params else False
        self.draw_sensor_range = gui_params['draw_sensor_range'] if gui_params else False
        self.zoom_level = gui_params['zoom_level'] if gui_params else False
        self.fix_vehicle_colors = False
        self.fix_pedestrian_colors = False
        self.fix_moped_colors =False
        self.gui_state_info = []
        self.gui_action_info = []
        self.gui_occlusions = []

        # Initialize state
        self.vehicles = []
        self.positions = np.zeros([self.max_nb_vehicles, 2])  # Defined as center of vehicle
        self.speeds = np.zeros([self.max_nb_vehicles, 2])
        self.accs = np.zeros([self.max_nb_vehicles])
        self.headings = np.zeros([self.max_nb_vehicles])
        self.ego_id = None

        # pedestrians state
        self.pedestrians = []
        self.ped_positions = np.zeros([self.max_nb_pedestrians, 2])
        self.ped_speeds = np.zeros([self.max_nb_pedestrians])
        self.ped_headings = np.zeros([self.max_nb_pedestrians])

        # moped states
        self.moped = []
        self.mop_positions = np.zeros([self.max_nb_moped, 2])
        self.mop_speeds = np.zeros([self.max_nb_moped, 2])
        self.mop_accs = np.zeros([self.max_nb_moped])
        self.mop_headings = np.zeros([self.max_nb_moped])

        self.previous_adding_node = None
        self.state_t0 = None
        self.state_t1 = None

        self.road = Road(road_params, start_time=start_time)
        # self.road.create_road()

        # To store history information
        self.hist_len = 10
        self.fut_len = 30
        self.hist_vehicles = np.zeros([self.max_nb_vehicles, self.hist_len, 5]) # features: x, y, vx, vy, heading
        self.hist_moped = np.zeros([self.max_nb_moped, self.hist_len, 5])
        self.hist_pedestrians = np.zeros([self.max_nb_pedestrians, self.hist_len, 5])
        self.hist_num = 0  # 
        
        self.nb_digits = int(np.floor(np.log10(self.max_nb_vehicles))) + 1
        self.nb_ped_digits = int(np.floor(np.log10(self.max_nb_pedestrians)))
        self.nb_mop_digits = int(np.floor(np.log10(self.max_nb_moped)))

        self.node_type_to_indicator_vec = { 'car': torch.tensor([[0,0,1]]),
                                    'truck': torch.tensor([[0,0,1]]),
                                    'bus': torch.tensor([[0,0,1]]),
                                    'motorcycle': torch.tensor([[0,0,1]]),
                                    'tricycle': torch.tensor([[0,0,1]]),
                                    'pedestrian': torch.tensor([[0,1,0]]),
                                    'bicycle': torch.tensor([[0,1,0]]),
                                    'map': torch.tensor([[1,0,0]]),
                                    'traffic_light': torch.tensor([[1,0,0]])}
        # Map information
        self.cur_map_png_path = '/home/kai/Vscode/SinD/SinD-main/visualizations/map_png/Tainjin_map.png'
        self.cur_map_path = '/home/kai/Vscode/SinD/SinD-main/visualizations/osm_maps/Tianjin.osm'
        self.map_img_np = self.read_img_to_numpy()
        self.map_limits_n_center = self.plot_map()       
        
        if self.use_gui:  
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        if sim_params['remove_sumo_warnings']:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start",
                         "--no-warnings"])
        else:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start"])

    def reset(self, ego_at_intersection=False, sumo_ctrl=False, eps=0):
        """
        Resets the intersection driving environment to a new random initial state.

        The ego vehicle starts in the south. A number of surrounding vehicles are added to random positions
        in the east and west, with randomly selected driver model parameters, e.g., desired speed.

        Args:
            ego_at_intersection (bool): If true, the ego vehicle starts close to the intersection (see the paper
                                        for description of specific tests)
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle.

        Returns:
            observation (ndarray): The observation of the traffic situation, according to the sensor model.
        """
        # Remove all vehicles
        # In some cases, the last vehicle may not yet have been inserted, and therefore cannot be deleted.
        # Then, run a few extra simulation steps.
        i = 0
        while not tuple(self.moped + self.vehicles) == traci.vehicle.getIDList():
            if i > 100:
                raise Exception("All vehicles could not be inserted, and therefore not reset.")
            if len(self.vehicles)+len(self.moped) - len(traci.vehicle.getIDList()) > 1:
                warnings.warn("More than one vehicle missing during reset")
            traci.simulationStep()
            i += 1

        i = 0
        while not tuple(self.pedestrians) == traci.person.getIDList():
            if i > 100:
                raise Exception("All pedestrians could not be inserted, and therefore not reset.")
            if len(self.pedestrians) - len(traci.person.getIDList()) > 1:
                warnings.warn("More than one person missing during reset")
            traci.simulationStep()
            i += 1

        for veh in self.vehicles:
            traci.vehicle.remove(veh)

        for ped in self.pedestrians:
            traci.person.removeStages(ped)
        
        for mop in self.moped:
            traci.vehicle.remove(mop)

        traci.simulationStep()

        # Reset vehicle state
        self.vehicles = []
        self.positions = np.zeros([self.max_nb_vehicles, 2])
        self.speeds = np.zeros([self.max_nb_vehicles, 2])
        self.accs = np.zeros([self.max_nb_vehicles])
        self.headings = np.zeros([self.max_nb_vehicles])

        # Reset pedestrians state
        self.pedestrians = []
        self.ped_positions = np.zeros([self.max_nb_pedestrians, 2])
        self.ped_speeds = np.zeros([self.max_nb_pedestrians])
        self.ped_headings = np.zeros([self.max_nb_pedestrians])

        # Reset moped state
        self.moped = []
        self.mop_positions = np.zeros([self.max_nb_moped, 2])
        self.mop_speeds = np.zeros([self.max_nb_moped, 2])
        self.mop_accs = np.zeros([self.max_nb_moped])
        self.mop_headings = np.zeros([self.max_nb_moped])

        self.previous_adding_node = None
        self.state_t0 = None
        self.state_t1 = None

        # historical information
        self.hist_vehicles = np.zeros([self.max_nb_vehicles, self.hist_len, 5]) # x, y, vx, vy, heading
        self.hist_moped = np.zeros([self.max_nb_moped, self.hist_len, 5])
        self.hist_pedestrians = np.zeros([self.max_nb_pedestrians, self.hist_len, 5])
        self.hist_num = 0

        # Add ego vehicle and determine its destination
        if self.goal == 'left-turn':
            self.start_route == 'route5'
        elif self.goal == 'go-straight':
            self.start_route == 'route9'
        else: # 'right-turn'
            self.start_lane == 'route6'

        self.ego_id = 'veh' + '0'.zfill(int(np.ceil(np.log10(self.max_nb_vehicles))))   # Add leading zeros to number
        traci.vehicle.add(self.ego_id, self.start_route, typeID='truck', depart=None, departLane=2,
                        departPos='base', departSpeed=self.road.road_params['vehicles'][0]['maxSpeed'],
                        arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                        line='', personCapacity=0, personNumber=0)
                          
        traci.vehicle.subscribe(self.ego_id, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, ANGLE])
        traci.simulationStep()

        if self.draw_sensor_range:
            traci.vehicle.highlight(self.ego_id, size=self.sensor_range)
        assert (len(traci.vehicle.getIDList()) == 1)
        self.vehicles = [self.ego_id]

        # Random init steps
        for i in range(self.init_steps - 1):
            self.step(action=[0, 0], sumo_ctrl=True)


        traci.vehicle.moveTo(self.ego_id, self.start_lane, 100)
        observation_ego, ego_light_state, observation_others, selected_agents, selected_interested_agents, done, info = self.step(action=[0, 0], sumo_ctrl=True, eps=eps)

        # # Turn off all internal lane changes and all safety checks for ego vehicle
        # if not sumo_ctrl:
        #     if not self.safety_check:
        #         traci.vehicle.setSpeedMode(self.ego_id, 0)
        #         traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        # else:
        #     traci.vehicle.setSpeed(self.ego_id, -1)

        # # Special case of ego vehicle starting close to the intersection
        # if ego_at_intersection:
        #     traci.vehicle.setSpeed(self.ego_id, 0)
        #     self.speeds[0, 0] = 7
        #     traci.vehicle.moveTo(self.ego_id, self.start_lane, -self.start_pos + self.max_ego_speed - self.lane_width
        #                          - self.occlusion_dist - self.speeds[0, 0])
        #     observation_ego, ego_light_state, observation_others, selected_agents, selected_interested_agents, done, info = self.step(action=[0, 0])

        self.step_ = 0

        if self.use_gui:
            traci.gui.setZoom('View #0', self.zoom_level)
            # self.draw_occlusion()
            if self.print_gui_info:
                self.print_state_info_in_gui(info='Start')

        return observation_ego, ego_light_state, observation_others, selected_agents, selected_interested_agents, done, info

    def step(self, action, action_info=None, sumo_ctrl=False, eps=0):
        """
        Transition the environment to the next state with the specified action.

        Args:
            action (float): MPCC outputs [acc, delta]
            action_info (dict): Used to display information in the GUI.
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle
                              (ignoring the surrounding vehicles).

        Returns:
            tuple, containing:

                observation_ego: state variables: [x, y, phi, v]; control variables: [acc, delta]

                observation_others (ndarray): the history information of selected traffic participants including current time,
                                                to feed to the prediction model.
                

                done (bool): True if terminal state is reached, otherwise False
                info (dict): Dictionary with simulation information.
        """
        episode = eps
        self.state_t0 = np.copy(self.state_t1)

        if self.use_gui and self.print_gui_info:
            self.print_action_info_in_gui(action, action_info)
        
        # Get traffic light states of the ego vehicle
        ego_light_all = traci.vehicle.getNextTLS(self.ego_id)
        ego_light_state = None
        if ego_light_all != ():
            light = ego_light_all[0]  # light = (tlsID, tlsIndex, distance, state)
            ego_light_state = light[3]

        global_traffic_light_state = traci.trafficlight.getRedYellowGreenState('1')

#----------------------dding traffic participants every second-------------------------#
        # Add more vehicles if possible
        if self.step_ % 10 ==0: # Cuz the frequency is set as 10hz
            # if np.random.rand()>= 0.25*0 or len(self.vehicles)<1:
            nb_vehicles = len(self.vehicles)
            if nb_vehicles < self.max_nb_vehicles:
                if np.random.rand() < self.adding_prob:
                    veh_id = 'veh' + str(nb_vehicles).zfill(int(np.ceil(np.log10(self.max_nb_vehicles))))  # Add leading zeros to number
                    if np.random.rand() < 0.2:
                        route_id = 3
                    else:
                        route_id = np.random.randint(10)
                        if route_id == 5 or route_id == 6 or route_id == 9: # to avoid collison with ego route
                             route_id =3
                    
                    if route_id in [0, 1]:
                        node = 0
                    elif route_id in [2, 7]:
                        node = 2
                    elif route_id in [3, 4, 8]:
                        node = 4
                    else:
                        node = 3
                        
                    while node == self.previous_adding_node:   # To avoid adding a vehicle to an already occupied spot
                        route_id = np.random.randint(10)
                        if route_id == 5 or route_id == 6 or route_id == 9: # to avoid collison with ego route
                             route_id =3
                        if route_id in [0, 1]:
                            node = 0
                        elif route_id in [2, 7]:
                            node = 2
                        elif route_id in [3, 4, 8]:
                            node = 4
                        else:
                            node = 3
                    self.previous_adding_node = node

                    speed = self.min_speed + np.random.rand() * (self.max_speed - self.min_speed)
                    if node == 4 or node == 3:
                        departLane = 2
                    else:
                        if route_id == 1 or route_id== 7:
                            departLane = 2
                        else:
                            departLane = np.random.randint(2, 4)
                    traci.vehicle.add(veh_id, 'route' + str(route_id), typeID='newvehicledist', depart=None, departLane=departLane,
                                    departPos='base', departSpeed=speed,
                                    arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                                    line='', personCapacity=0, personNumber=0)
                    traci.vehicle.setMaxSpeed(veh_id, speed)
                    traci.vehicle.setLaneChangeMode(veh_id, 0)   # Turn off lane changes
                    traci.vehicle.subscribe(veh_id, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, ANGLE])  # position, speed
                    self.vehicles.append(veh_id)
            # if np.random.rand()>=0.7*0 or len(self.pedestrians)<1:
                # Add more pedestrians if possible
            nb_pedestrians = len(self.pedestrians)
            if nb_pedestrians < self.max_nb_pedestrians:
                ped_id = 'ped' + str(nb_pedestrians).zfill(int(np.ceil(np.log10(self.max_nb_pedestrians))))# Add leading zeros to number
                speed = np.random.normal(1.38, 0.28) ###行人速度--服从正态分布
                if np.random.rand() <= self.ped_adding_prob: ###每个step添加行人的概率
                    traci.person.add(ped_id, edgeID='L12', pos=20, typeID="pedestrians")  # east to west lower
                    traci.person.appendWalkingStage(ped_id, ["L12", "L01", "L50"], 2000)
                    # print('east to west lower')
                elif np.random.rand() > self.ped_adding_prob and np.random.rand() <= self.ped_adding_prob*2:
                    traci.person.add(ped_id, edgeID='L21', pos=180, typeID="pedestrians")  # east to west upper
                    traci.person.appendWalkingStage(ped_id, ["L21", "L10", "L05"], 2000)
                    # print('east to west upper')
                elif np.random.rand() > self.ped_adding_prob*2 and np.random.rand() <= self.ped_adding_prob*3:
                    traci.person.add(ped_id, edgeID='L31', pos=180, typeID="pedestrians")  # south to north right
                    traci.person.appendWalkingStage(ped_id, ["L31", "L14", "L48"], 2000)
                    # print('south to north right')
                else:
                    traci.person.add(ped_id, edgeID='L41', pos=180, typeID="pedestrians")  # north to sourth left
                    traci.person.appendWalkingStage(ped_id, ["L41", "L13", "L37"], 2000)
                    # print('north to sourth left')
    
                traci.person.setSpeed(ped_id, speed)#设置行人速度
                traci.person.subscribe(ped_id, [POSITION, LONG_SPEED, ANGLE])
                self.pedestrians.append(ped_id)
            # if np.random.rand()>0.9*0 or len(self.moped)<1:
            # Add Moped if possible
            nb_moped = len(self.moped)
            if nb_moped < self.max_nb_moped:
                # if np.random.rand() < self.mop_adding_prob:
                mop_id = 'mop' + str(nb_moped).zfill(int(np.ceil(np.log10(self.max_nb_moped))))  # Add leading zeros to number
                route_id = [0, 2, 3][np.random.randint(3)]
                speed = self.min_speed + np.random.rand() * (self.max_speed - self.min_speed)*0.45
                traci.vehicle.add(mop_id, 'route' + str(route_id), typeID='moped', depart=None, departLane=1,
                                departPos='base', departSpeed=speed,
                                arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                                line='', personCapacity=0, personNumber=0)
                traci.vehicle.setMaxSpeed(mop_id, speed)
                traci.vehicle.setLaneChangeMode(mop_id, 0)   # Turn off lane changes
                traci.vehicle.subscribe(mop_id, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, ANGLE])  # position, speed
                self.moped.append(mop_id)
#--------------------------adding traffic participants every second-------------------------#

#-----------------------------take one simulation step--------------------------------------#

        # extract the ego vehicle states for updating 
        out = traci.vehicle.getSubscriptionResults(self.ego_id)
        ego_velocity = np.sqrt(out[LONG_SPEED]*out[LONG_SPEED] + out[LAT_SPEED]*out[LAT_SPEED])

        ego_heading = np.mod(-out[ANGLE]+90, 360)/180*np.pi
        ego_x = out[POSITION][0] - 4000 - self.car_length / 2 * np.cos(ego_heading)
        ego_y = out[POSITION][1] - 4000 - self.car_length / 2 * np.sin(ego_heading)
        # print('xx', ego_x-(ego_x- self.ego_length / 2* np.cos(ego_heading)))
        # print('yy', ego_y-(ego_y- self.ego_length / 2* np.sin(ego_heading)))
        z = np.array([action[0], action[1], ego_x, ego_y, ego_heading, ego_velocity])


        updated_states = np.array(self.equation(z))

        updated_ego_x = updated_states[0] + 4000 + self.car_length / 2 * np.cos(ego_heading)
        updated_ego_y = updated_states[1] + 4000 + self.car_length / 2 * np.sin(ego_heading)
        updated_ego_heading = 90-updated_states[2]/np.pi*180 # sumo degree
        uddated_ego_velocity = updated_states[3]

        # send the control command to SUMO simulator
        if not sumo_ctrl:
            # traci.vehicle.setSpeed(self.ego_id, uddated_ego_velocity) #edgeID='L31', lane=2
            traci.vehicle.moveToXY(vehID=self.ego_id, edgeID='dummy', lane=-1, x=updated_ego_x[0], y=updated_ego_y[0], angle=updated_ego_heading[0], keepRoute=2)
            # traci.vehicle.moveToXY(vehID=self.ego_id, edgeID='dummy', lane=-1, x=4000, y=4000, angle=0, keepRoute=2)
        traci.simulationStep()
        self.step_ += 1
#-----------------------------take one simulation step---------------------------------------#

#-------------------------extract environment information------------------------------------#
        # Get state information
        
        for veh in self.vehicles:
            i = int(veh[-self.nb_digits:])
            out = traci.vehicle.getSubscriptionResults(veh)
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED]
            self.accs[i] = out[LONG_ACC]
            self.headings[i] = np.mod(-out[ANGLE]+90, 360)/180*np.pi # math degree
            if i == 0:
                vehicle_length = self.ego_length
            else:
                vehicle_length = self.car_length
            self.positions[i, 0] = out[POSITION][0] - vehicle_length / 2 * np.cos(self.headings[i])-4000 # convert the central point as [0, 0]
            self.positions[i, 1] = out[POSITION][1] - vehicle_length / 2 * np.sin(self.headings[i])-4000
            
            # store the states in historical docker
            hist_num = sum(self.hist_vehicles[i, :, 0]!=np.zeros(self.hist_len)) # check the number of existing historic frames

            if sum(self.hist_vehicles[i, -1] == np.zeros(5))==5: # check the docker is fully filled
                self.hist_vehicles[i, hist_num] = [self.positions[i,0], self.positions[i,1], self.speeds[i,0]*np.cos(self.headings[i]), self.speeds[i,0]*np.sin(self.headings[i]), self.headings[i]]
            else:
                temp = self.hist_vehicles[i,1:,:]
                self.hist_vehicles[i, 0:-1, :] = temp
                self.hist_vehicles[i, -1, :] = [self.positions[i,0], self.positions[i,1], self.speeds[i,0]*np.cos(self.headings[i]), self.speeds[i,0]*np.sin(self.headings[i]), self.headings[i]]

        # Get pedestrian information
        
        for ped in self.pedestrians:
            j = int(ped[-self.nb_ped_digits:])
            ped_out = traci.person.getSubscriptionResults(ped)
            self.ped_speeds[j] = ped_out[LONG_SPEED] # sumo doesn't provide lateral speed, it is set as 0.
            self.ped_headings[j] = np.mod(-ped_out[ANGLE]+90, 360)/180*np.pi
            self.ped_positions[j, 0] = ped_out[POSITION][0]-4000
            self.ped_positions[j, 1] = ped_out[POSITION][1]-4000

            # store the states in historical docker
            ped_hist_num = sum(self.hist_pedestrians[j, :, 0]!=np.zeros(self.hist_len)) # check the number of existing historic frames
            if sum(self.hist_pedestrians[j, -1] ==np.zeros(5))==5: # check the docker is fully filled
                self.hist_pedestrians[j, ped_hist_num] = [self.ped_positions[j,0], self.ped_positions[j,1], self.ped_speeds[j]*np.cos(self.ped_headings[j]), self.ped_speeds[j]*np.sin(self.ped_headings[j]), self.ped_headings[j]]
            else:
                temp = self.hist_pedestrians[j, 1:] # to delete the first value.
                # self.hist_pedestrians[j] = np.zeros([self.hist_len, 5])
                self.hist_pedestrians[j, 0:-1, :] = temp
                self.hist_pedestrians[j, -1,:] = [self.ped_positions[j,0], self.ped_positions[j,1], self.ped_speeds[j]*np.cos(self.ped_headings[j]), self.ped_speeds[j]*np.sin(self.ped_headings[j]), self.ped_headings[j]]
        
        # Get moped state information
        for mop in self.moped:
            k = int(mop[-self.nb_mop_digits:])
            mop_out = traci.vehicle.getSubscriptionResults(mop)
            self.mop_speeds[k, 0] = mop_out[LONG_SPEED]
            self.mop_speeds[k, 1] = mop_out[LAT_SPEED]
            self.mop_accs[k] = mop_out[LONG_ACC]
            self.mop_headings[k] = np.mod(-mop_out[ANGLE]+90, 360)/180*np.pi
            self.mop_positions[k, 0] = mop_out[POSITION][0] - self.moped_length / 2 * np.cos(self.mop_headings[k])-4000
            self.mop_positions[k, 1] = mop_out[POSITION][1] - self.moped_length / 2 * np.sin(self.mop_headings[k])-4000

            # store the states in historical docker
            mop_hist_num = sum(self.hist_moped[k, 0:self.hist_len, 0]!=np.zeros(self.hist_len)) # check the number of existing historic frames

            if sum(self.hist_moped[k, self.hist_len-1] ==np.zeros(5))==5: # check the docker is fully filled
                self.hist_moped[k, mop_hist_num] = [self.mop_positions[k, 0], self.mop_positions[k,1], self.mop_speeds[k,0]*np.cos(self.mop_headings[k]), self.mop_speeds[k,0]*np.sin(self.mop_headings[k]), self.mop_headings[k]]
            else:
                temp = self.hist_moped[k,1:] # to delete the first value.
                # self.hist_moped[k] = np.zeros([self.hist_len, 5])
                self.hist_moped[k, 0:-1, :] = temp
                self.hist_moped[k, -1, :] = [self.mop_positions[k, 0], self.mop_positions[k,1], self.mop_speeds[k,0]*np.cos(self.mop_headings[k]), self.mop_speeds[k,0]*np.sin(self.mop_headings[k]), self.mop_headings[k]]
#-----------------------------extract environment information--------------------------------#

        # To contruct pyg data for the prediction model
        observation_others, selected_agents = self.get_pyg_data(light_state=global_traffic_light_state)
        observation_ego = updated_states
                
        # if traci.simulation.getCollidingVehiclesNumber() > 0:
        #     warnings.warn('Collision between surrounding cars. This should normally not happen.')

        # Check terminal state
        info = {}
        done = False

        collision, near_collision, collision_info = self.collision_detection()

        if collision:
            done = True
            info['terminal_reason'] = str(collision_info)

        if self.step_ == self.max_steps:
            done = True
            info['terminal_reason'] = 'Max steps'

        # if np.abs(self.positions[0, 0] - self.intersection_pos[0]) >= self.end_pos:
        if math.sqrt((self.positions[0, 0]-self.end_pos[0])**2+(self.positions[0, 1]-self.end_pos[1])**2) < 1:
            done = True
            info['terminal_reason'] = 'Success'
        
        if self.use_gui and self.print_gui_info:
            
            self.print_state_info_in_gui(info=info, episode=episode)
        
        # select interested agents to be considered in MPC, because they are too much agents at intersection
        # and some of them don't need to be considered.
        # note that the rotation angle needs to substract pi/2.

        Rotation_ego = np.array([[np.cos(self.headings[0]-np.pi/2), -np.sin(self.headings[0]-np.pi/2)],
                        [np.sin(self.headings[0]-np.pi/2), np.cos(self.headings[0]-np.pi/2)]])
        Rotate_veh = np.dot(self.positions[0:len(self.vehicles)]-self.positions[0], Rotation_ego)
        Rotate_mop = np.dot(self.mop_positions[0:len(self.moped)]-self.positions[0], Rotation_ego)
        Rotate_ped = np.dot(self.ped_positions[0:len(self.pedestrians)]-self.positions[0], Rotation_ego)

 
        # only other vehicles in the front the ego vehilce will be considered in MPC.
        interested_vehicle= np.array([np.linalg.norm(self.positions[0:len(self.vehicles)] - self.positions[0], axis=1) <= self.sensor_range,
                    np.logical_not(np.array([self.positions[0:len(self.vehicles), 0] == 0, self.positions[0:len(self.vehicles), 1] == 0]).all(0)),
                        self.positions[0:len(self.vehicles), 0] <= 50, self.positions[0:len(self.vehicles), 0] >= -50,
                        self.positions[0:len(self.vehicles), 1] <= 50, self.positions[0:len(self.vehicles), 1] >= -50,
                        self.speeds[0:len(self.vehicles), 0] > 1,
                        Rotate_veh[0:len(self.vehicles), 1] > 0,
                        np.abs(np.divide(Rotate_veh[0:len(self.vehicles), 1], Rotate_veh[0:len(self.vehicles), 0], 
                        10*np.ones_like(Rotate_veh[0:len(self.vehicles), 1]),where=Rotate_veh[0:len(self.vehicles), 0]!=0))>=np.tan(np.pi/3)]).all(0)

        selected_interested_vehicle = np.array(self.vehicles)[interested_vehicle]

        interested_mop = np.array([np.linalg.norm(self.mop_positions[0:len(self.moped)] - self.positions[0], axis=1) <= self.sensor_range,
                    np.logical_not(np.array([self.mop_positions[0:len(self.moped), 0] == 0, self.mop_positions[0:len(self.moped), 1] == 0]).all(0)),
                    self.mop_positions[0:len(self.moped), 0] <= 50, self.mop_positions[0:len(self.moped), 0] >= -50,
                    self.mop_positions[0:len(self.moped), 1] <= 50, self.mop_positions[0:len(self.moped), 1] >= -50,
                    self.mop_speeds[0:len(self.moped), 0] > 1,
                    Rotate_mop[0:len(self.moped), 1] > 0,
                    np.abs(np.divide(Rotate_mop[0:len(self.moped), 1], Rotate_mop[0:len(self.moped), 0], 
                    10*np.ones_like(Rotate_mop[0:len(self.moped), 1]), where=Rotate_mop[0:len(self.moped), 0]!=0))>=np.tan(np.pi/3)]).all(0)

        selected_interested_mop = np.array(self.moped)[interested_mop]

        interested_ped = np.array([np.linalg.norm(self.ped_positions[0:len(self.pedestrians)] - self.positions[0], axis=1) <= self.sensor_range,
                    np.logical_not(np.array([self.ped_positions[0:len(self.pedestrians), 0] == 0, self.ped_positions[0:len(self.pedestrians), 1] == 0]).all(0)),
                    self.ped_positions[0:len(self.pedestrians), 0] <= 50, self.ped_positions[0:len(self.pedestrians), 0] >= -50,
                    self.ped_positions[0:len(self.pedestrians), 1] <= 50, self.ped_positions[0:len(self.pedestrians), 1] >= -50,
                    self.ped_speeds[0:len(self.pedestrians)]>0,
                    Rotate_ped[0:len(self.pedestrians), 1] > 0,
                    np.abs(np.divide(Rotate_ped[0:len(self.pedestrians), 1], Rotate_ped[0:len(self.pedestrians), 0],
                    10*np.ones_like(Rotate_ped[0:len(self.pedestrians), 1]), where=Rotate_ped[0:len(self.pedestrians), 0]!=0))>=np.tan(np.pi/3)]).all(0)
        selected_interested_ped = np.array(self.pedestrians)[interested_ped]

        selected_interested_agents = selected_interested_vehicle.tolist()+selected_interested_mop.tolist()+selected_interested_ped.tolist()

        # print('selected_interested_agents:',selected_interested_agents)

        return observation_ego, ego_light_state, observation_others, selected_agents, selected_interested_agents, done, info
    
    def equation(self, z):
        # Because SUMO doesn't provide any vehicle model that take the steering angle as input, 
        # so we utilize an ODEs depicting vehilce dynamics and use Runge-Kutta integrator to obtain next states
        # Next, use traci.vehicle.moveToXY to show the trajectory in SUMO.
        integrator_stepsize = 0.1
        return forcespro.nlp.integrate(vehicle_model, z[2:6], z[0:2],
                                                    integrator=forcespro.nlp.integrators.RK4,
                                                    stepsize=integrator_stepsize)

    # construct a pyg data for the prediction model.
    def get_pyg_data(self, light_state):

        # decide which vehicles are predicted. 
        vehicle_in_range = np.array([self.positions[0:len(self.vehicles), 0] <= 50, self.positions[0:len(self.vehicles), 0] >= -50,
                    self.positions[0:len(self.vehicles), 1] <= 50, self.positions[0:len(self.vehicles), 1] >= -50]).all(0)

        selected_vehicle_in_range = np.array(self.vehicles)[vehicle_in_range]

        mop_in_range = np.array([self.mop_positions[0:len(self.moped), 0] <= 50, self.mop_positions[0:len(self.moped), 0] >= -50,
                    self.mop_positions[0:len(self.moped), 1] <= 50, self.mop_positions[0:len(self.moped), 1] >= -50]).all(0)

        selected_mop_in_range = np.array(self.moped)[mop_in_range]

        ped_in_range = np.array([self.ped_positions[0:len(self.pedestrians), 0] <= 50, self.ped_positions[0:len(self.pedestrians), 0] >= -50,
                    self.ped_positions[0:len(self.pedestrians), 1] <= 50, self.ped_positions[0:len(self.pedestrians), 1] >= -50]).all(0)
        
        selected_ped_in_range = np.array(self.pedestrians)[ped_in_range]

        selected_agents = selected_vehicle_in_range.tolist()+selected_mop_in_range.tolist()+selected_ped_in_range.tolist()

        raw_v_IDs =selected_agents

        v_id_to_node_index = {}

        for i, v_id in enumerate(raw_v_IDs):  # Assign the Node_ID for all agents in the current frame . 
            v_id_to_node_index[v_id] = i
        
        # Node features for all agents in the current frame
        Nodes_f = torch.zeros((len(raw_v_IDs), self.hist_len, 4)) # [x, y, vx, vy].
        Fut_GT = torch.zeros((len(raw_v_IDs), self.fut_len, 2)) # to fit the format of the prediction model.

        # Edge features
        Edges = torch.empty((2, 0)).long()
        Edges_attr = torch.empty((5, 0)) # [d_x, d_y, d_vx, d_vy, d_psi]
        Edges_type = torch.empty((6, 0)).long()

        # Veh-map features
        Veh_Map_Attr = torch.empty((5, 0))

        # Traffic light features
        Veh_Traffic_light_Attr = torch.empty((4, 0))
        
        # Ground turth
        GT = torch.zeros((len(raw_v_IDs), self.fut_len, 2))
        
        # Masks
        Tar_Mask = []
        Veh_Tar_Mask = []
        Veh_Mask = []
        Ped_Mask = []

        # set map features
        Map = torch.from_numpy(np.copy(self.map_img_np))
        Map_center = torch.tensor(self.map_limits_n_center['map_center'])

        # Raw hist and fut
        Raw_hist = torch.zeros((len(raw_v_IDs), self.hist_len, 4)) 
        Raw_fut = torch.zeros((len(raw_v_IDs), self.fut_len, 2))
        Raw_heading = torch.zeros((len(raw_v_IDs),1))

        for i, v_id in enumerate(raw_v_IDs):

            # Node features
            v_hist, raw_h, agn_cur_psi_rad = self.get_hist(agn_id=v_id)

            Nodes_f[i] = torch.from_numpy(v_hist) # convert numpy.ndarray to tensor
            Raw_hist[i] = torch.from_numpy(raw_h)

            Raw_heading[i] = torch.from_numpy(agn_cur_psi_rad)
             
            if v_id[0:3] == 'veh':
                ind = int(v_id[-self.nb_digits:])
                agn_cur_state = np.array([self.positions[ind, 0], self.positions[ind, 1], self.speeds[ind, 0]*np.cos(self.headings[ind]), self.speeds[ind, 0]*np.sin(self.headings[ind]), self.headings[ind]])
                agn_type = 'car'
                Veh_Mask.append(True)
                Ped_Mask.append(False)
            elif v_id[0:3] == 'mop':
                ind = int(v_id[-self.nb_mop_digits:])
                agn_cur_state = np.array([self.mop_positions[ind, 0], self.mop_positions[ind, 1], self.mop_speeds[ind, 0]*np.cos(self.mop_headings[ind]), self.mop_speeds[ind, 0]*np.sin(self.mop_headings[ind]), self.mop_headings[ind]])
                agn_type = 'motorcycle'
                Veh_Mask.append(True)
                Ped_Mask.append(False)
            else:
                ind = int(v_id[-self.nb_ped_digits:])
                agn_cur_state = np.array([self.ped_positions[ind, 0], self.ped_positions[ind, 1], self.ped_speeds[ind]*np.cos(self.ped_headings[ind]), self.ped_speeds[ind]*np.sin(self.ped_headings[ind]), self.ped_headings[ind]])
                agn_type = 'pedestrian'
                Veh_Mask.append(False)
                Ped_Mask.append(True)

            agn_cur_state = torch.from_numpy(agn_cur_state)

            v_nbrs = self.get_nbrs(agn_id=v_id, radii=30)

            v_node_index = v_id_to_node_index[v_id]

            self_edge = torch.tensor([[v_node_index], [v_node_index]])

            self_edge_attr = torch.tensor([0, 0, 0, 0, 0]).float().unsqueeze(dim=1)

            self_edge_type = torch.cat((self.node_type_to_indicator_vec[agn_type], self.node_type_to_indicator_vec[agn_type]), dim=1)

            Edges = torch.cat((Edges, self_edge), dim=1)

            Edges_attr = torch.cat((Edges_attr, self_edge_attr), dim=1)

            Edges_type = torch.cat((Edges_type, self_edge_type.transpose(0,1)), dim=1)
            
            # print('Edges_attr', Edges_attr)

            for v_nbr_id in v_nbrs:
                nbr_v_node_index = v_id_to_node_index[v_nbr_id]
                if v_nbr_id[0:3] == 'veh':
                    ind = int(v_nbr_id[-self.nb_digits:])
                    nbr_cur_state = np.array([self.positions[ind, 0], self.positions[ind, 1], self.speeds[ind, 0]*np.cos(self.headings[ind]), self.speeds[ind, 0]*np.sin(self.headings[ind]), self.headings[ind]])
                    nbr_type = 'car'
 
                elif v_nbr_id[0:3] == 'mop':
                    ind = int(v_nbr_id[-self.nb_mop_digits:])
                    nbr_cur_state = np.array([self.mop_positions[ind, 0], self.mop_positions[ind, 1], self.mop_speeds[ind, 0]*np.cos(self.mop_headings[ind]), self.mop_speeds[ind, 0]*np.sin(self.mop_headings[ind]), self.mop_headings[ind]])
                    nbr_type = 'motorcycle'
 
                else:
                    ind = int(v_nbr_id[-self.nb_ped_digits:])
                    nbr_cur_state = np.array([self.ped_positions[ind, 0], self.ped_positions[ind, 1], self.ped_speeds[ind]*np.cos(self.ped_headings[ind]), self.ped_speeds[ind]*np.sin(self.ped_headings[ind]), self.ped_headings[ind]])
                    nbr_type = 'pedestrian'
                nbr_cur_state = torch.from_numpy(nbr_cur_state)

                # edge
                edge = torch.tensor([[nbr_v_node_index], [v_node_index]])

                edge_attr = nbr_cur_state - agn_cur_state

                edge_attr = edge_attr.float().unsqueeze(dim=1)

                edge_type = torch.cat((self.node_type_to_indicator_vec[nbr_type], self.node_type_to_indicator_vec[agn_type]), dim=1)

                Edges = torch.cat((Edges, edge), dim=1)

                Edges_attr = torch.cat((Edges_attr, edge_attr), dim=1)

                Edges_type = torch.cat((Edges_type, edge_type.transpose(0,1)), dim=1)

            veh_map_attr = agn_cur_state.float() - torch.cat((Map_center.float(), torch.tensor([0., 0., 0.])), dim=0)

            veh_map_attr = veh_map_attr.unsqueeze(dim=1)

            Veh_Map_Attr = torch.cat((Veh_Map_Attr, veh_map_attr), dim=1)

            # Current traffic lights states
            traffic_light_cur_state = torch.tensor([0, 0, 0, 0]).float().unsqueeze(dim=1) # red:0; green: 1; yellow: 3
            for tl in [1, 2, 3, 4]:
                temp_traffic_light = light_state[(tl-1)*7]

                if temp_traffic_light == 'r':
                    traffic_light_cur_state[tl-1] = 0
                elif temp_traffic_light == 'y':
                    traffic_light_cur_state[tl-1] = 3
                else:
                    traffic_light_cur_state[tl-1] = 1
        
            Veh_Traffic_light_Attr = torch.cat((Veh_Traffic_light_Attr, traffic_light_cur_state), dim=1)

            tar = True
            Tar_Mask.append(tar)
            if agn_type == 'car' or agn_type =='motorcycle' and  tar:
                Veh_Tar_Mask.append(True)
            else:
                Veh_Tar_Mask.append(False)
            
        Tar_Mask = torch.tensor(Tar_Mask) # 未来轨迹是否满足条件 True
        Veh_Tar_Mask = torch.tensor(Veh_Tar_Mask) # 类型类型维car 且 未来轨迹满足条件
        Veh_Mask = torch.tensor(Veh_Mask) # 类型为car
        Ped_Mask = torch.tensor(Ped_Mask) # 类型为行人或者自行车

        pyg_data = Data(x=Nodes_f, y=Fut_GT, 
            edge_index=Edges, edge_attr=Edges_attr.transpose(0,1), edge_type=Edges_type.transpose(0,1), veh_map_attr=Veh_Map_Attr.transpose(0,1),
            tar_mask=Tar_Mask, veh_tar_mask=Veh_Tar_Mask, veh_mask=Veh_Mask, ped_mask=Ped_Mask,
            raw_hists=Raw_hist, raw_futs=Raw_fut, tra_lig=Veh_Traffic_light_Attr.t().float(), raw_headings=Raw_heading)

        return pyg_data, selected_agents
    
    def get_hist(self, agn_id, radii=30):
        if agn_id[0:3] == 'veh':
            ind = int(agn_id[-self.nb_digits:])
            raw_hist = self.hist_vehicles[ind][0:self.hist_len, 0:4]
            agn_cur_pos = self.positions[ind]
            agn_cur_psi_rad = self.headings[ind]
            
        elif agn_id[0:3] == 'mop':
            ind = int(agn_id[-self.nb_mop_digits:])
            raw_hist = self.hist_moped[ind][0:self.hist_len, 0:4]
            agn_cur_pos = self.mop_positions[ind]
            agn_cur_psi_rad = self.mop_headings[ind]
        else:
            ind = int(agn_id[-self.nb_ped_digits:])
            raw_hist = self.hist_pedestrians[ind][0:self.hist_len, 0:4]
            agn_cur_pos = self.ped_positions[ind]
            agn_cur_psi_rad = self.ped_headings[ind]

        new_hist = raw_hist - np.insert(agn_cur_pos, 2, [0, 0])

        new_hist[:,0], new_hist[:,1] = self.Srotate(agn_cur_psi_rad, new_hist[:,0], new_hist[:,1], 0, 0) # Rotated position w.r.t. the agent itself
        new_hist[:,2], new_hist[:,3] = self.Srotate(agn_cur_psi_rad, new_hist[:,2], new_hist[:,3], 0, 0) 
    
        return new_hist, raw_hist, np.array([agn_cur_psi_rad])
    
    def get_nbrs(self, agn_id, radii=30):
        
        if agn_id[0:3] == 'veh':
            
            ind = int(agn_id[-self.nb_digits:])
            ref_pos = self.positions[ind]
            
        elif agn_id[0:3] == 'mop':
            
            ind = int(agn_id[-self.nb_mop_digits:])
            ref_pos = self.mop_positions[ind]
        else:
            ind = int(agn_id[-self.nb_ped_digits:])
            ref_pos = self.ped_positions[ind]
        
        num_veh = len(self.vehicles)
        num_mop = len(self.moped)
        num_ped = len(self.pedestrians)

        # vehicle
        x_sqr = np.square(self.positions[0:num_veh, 0]-ref_pos[0])
        y_sqr = np.square(self.positions[0:num_veh, 1]-ref_pos[1])
        dis2tar = np.sqrt(x_sqr + y_sqr)

        veh_index = np.where((dis2tar<30)&(dis2tar>0.01)&(self.positions[0:num_veh, 0]<50)&(self.positions[0:num_veh, 0]>-50)\
            &(self.positions[0:num_veh, 1]<50)&(self.positions[0:num_veh, 1]>-50))

        veh_nbrs = np.array(self.vehicles)[veh_index].tolist()

        # mop
        x_sqr = np.square(self.mop_positions[0:num_mop, 0]-ref_pos[0])
        y_sqr = np.square(self.mop_positions[0:num_mop, 1]-ref_pos[1])
        dis2tar = np.sqrt(x_sqr + y_sqr)

        mop_index = np.where((dis2tar<30)&(dis2tar>0.01)&(self.mop_positions[0:num_mop, 0]<50)&(self.mop_positions[0:num_mop, 0]>-50)\
            &(self.mop_positions[0:num_mop, 1]<50)&(self.mop_positions[0:num_mop, 1]>-50))

        mop_nbrs = np.array(self.moped)[mop_index].tolist()

        # pedstrian
        x_sqr = np.square(self.ped_positions[0:num_ped, 0]-ref_pos[0])
        y_sqr = np.square(self.ped_positions[0:num_ped, 1]-ref_pos[1])
        dis2tar = np.sqrt(x_sqr + y_sqr)

        ped_index = np.where((dis2tar<30)&(dis2tar>0.01)&(self.ped_positions[0:num_ped, 0]<50)&(self.ped_positions[0:num_ped, 0]>-50)\
            &(self.ped_positions[0:num_ped, 1]<50)&(self.ped_positions[0:num_ped, 1]>-50))

        ped_nbrs = np.array(self.pedestrians)[ped_index].tolist()

        return veh_nbrs+mop_nbrs+ped_nbrs

    def Srotate(self, angle, valuex, valuey, pointx, pointy):
 
        sRotatex = (valuex-pointx)*np.cos(angle) + (valuey-pointy)*np.sin(angle) + pointx
        sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
        return sRotatex, sRotatey
    
    def read_img_to_numpy(self): # 
        '''return:gray_image'''
        with Image.open(self.cur_map_png_path) as map_png_image:
            gray_image = ImageOps.grayscale(map_png_image)
        return np.asarray(gray_image)
    def plot_map(self):
        fig, axes = plt.subplots(1, 1)
        # plt.subplots_adjust(wspace=0, hspace=0)/
        fig.canvas.set_window_title("SUMO Dataset Visualization")
        lat_origin, lon_origin = 0. , 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        map_vis_xy.draw_map_without_lanelet_xy(self.cur_map_path, axes, lat_origin, lon_origin)

        x_ax_limits, y_ax_limits = axes.get_xlim(), axes.get_ylim()
        plt.close()
        map_center_x = 0.5 * (x_ax_limits[1] + x_ax_limits[0])  # 地图中心的坐标
        map_center_y = 0.5 * (y_ax_limits[1] + y_ax_limits[0])
        return {'map_xlim': x_ax_limits, 'map_ylim': y_ax_limits, 'map_center': [map_center_x, map_center_y]}

    def collision_detection(self):
        """ Only works for this specific case with crossing traffic, simplified for speed of execution.
        """
        # Check if ego vehicle violate the traffic light rules.
        light_all = traci.vehicle.getNextTLS(self.ego_id)  # traffic lights for the ego vehicle
        if light_all != ():
            light = light_all[0]  # light = (tlsID, tlsIndex, distance, state)
            light_id = light[0]
            # traffic light states
            light_state = light[3]
            light_info = traci.trafficlight.getAllProgramLogics(light_id)  
            logic_list = light_info[0]
            phase_info = logic_list.phases
            now_time = traci.simulation.getTime()
            switch_time = traci.trafficlight.getNextSwitch(light_id) # switch time
            sign = traci.trafficlight.getRedYellowGreenState(light_id) # phrase
            for t in range(len(phase_info)):
                phase_now = phase_info[t]
                phase_now_state = phase_now.state
                if sign == phase_now_state : ##确定了目前所处阶段
                    current_phase = t
            if current_phase == 0:
                switch_time_remain = switch_time- now_time + 3 
            else:
                switch_time_remain = switch_time - now_time  

            if light_state == 'r' :
                if ((self.intersection_pos[1]-4000 - self.stop_line) - (self.positions[0][1] + self.ego_length/2)) <= 0.1 \
                    and ((self.intersection_pos[1]-4000 - self.stop_line) - (self.positions[0][1] + self.ego_length/2)) >= 0 and switch_time_remain != 0:
                    # print('111', ((self.intersection_pos[1]-4000 - self.stop_line) - (self.positions[0][1] + self.ego_length)))
                    return True, False, 'Red light running'

            if light_state == 'y' :
                if ((self.intersection_pos[1]-4000 - self.stop_line) - (self.positions[0][1] + self.ego_length/2)) <=0.5\
                    and ((self.intersection_pos[1]-4000 - self.stop_line) - (self.positions[0][1] + self.ego_length/2)) > 0:
                    return False, False, 'Yellow light running'
        
        rotated = np.zeros([len(self.vehicles), 2, 4])
        for ind in range(len(self.vehicles)):
            rotated[ind, :, :] = rotate_box(self.headings[ind], self.positions[ind,:], 4.5, 1.5)

        mop_rotated = np.zeros([len(self.moped), 2, 4])
        for ind in range(len(self.moped)):
            mop_rotated[ind, :, :] = rotate_box(self.mop_headings[ind], self.mop_positions[ind], 2.8, 0.8)
        
        ped_rotated = np.zeros([len(self.pedestrians), 2, 4])
        for ind in range(len(self.pedestrians)):
            ped_rotated[ind, :, :] = rotate_box(self.ped_headings[ind], self.ped_positions[ind], 0.35, 0.5)

        flags = []
        for index in range(1,len(self.vehicles)):
            if len(self.vehicles) == 1:
                continue
            else:
                flags.append(intersect_detection(rotated[index], rotated[0]))
        mop_flags = []    
        for index in range(len(self.moped)):
            mop_flags.append(intersect_detection(mop_rotated[index], rotated[0]))
        ped_flags = []    
        for index in range(len(self.pedestrians)):
            ped_flags.append(intersect_detection(ped_rotated[index], rotated[0]))
        
        # print('flags', flags+mop_flags+ped_flags)

        if sum(flags) > 0:
            return True, False, 'collision with vehicle'
        elif sum(mop_flags) >0:
            return True, False, 'collision with mop'
        elif sum(ped_flags) >0:
            return True, False, 'collision with pedestrian'
        else:
            return False, False, None


    def print_state_info_in_gui(self, info=None, episode=0):
        """
        Prints information in the GUI.
        """
        for item in self.gui_state_info:
            traci.polygon.remove(item)
        dy = 15
        
        self.gui_state_info = ['Position: [{:.2f},{:-.2f}]'.format(self.positions[0, 0], self.positions[0, 1]),
                               'Speed: {0:.1f}'.format(self.speeds[0, 0]),
                               'Collision: ' +  str(info),
                                'Step: ' + str(self.step_),
                                'Episode: ' + str(episode)]
        for idx, text in enumerate(self.gui_state_info):
            traci.polygon.add(text, [self.road.road_params['info_pos'],
                                     self.road.road_params['info_pos'] + [1, -idx*dy]], [0, 0, 0, 0])


    def print_action_info_in_gui(self, action=None, action_info=None):
        """
        Prints information in the GUI.
        """
        # if action == 0:
        #     action_str = 'cruise'
        # elif action == 1:
        #     action_str = 'go'
        # elif action == 2:
        #     action_str = 'stop'
        # else:
        #     action_str = 'backup'
        # for item in self.gui_action_info:
        #     traci.polygon.remove(item)
        # dy = 15
        # self.gui_action_info = ['Action: ' + action_str]
        # traci.polygon.add(self.gui_action_info[0],
        #                   [self.road.road_params['info_pos'],  self.road.road_params['info_pos'] + [1, dy]],
        #                   [0, 0, 0, 0])
        # if action_info is not None:

        #     if 'q_values' in action_info:
        #         self.gui_action_info.append('                                  cruise  |    go      |   stop  ')
        #         traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
        #                                                      self.road.road_params['action_info_pos'] + [1, 1*dy]],
        #                           [0, 0, 0, 0])
        #         self.gui_action_info.append('Q-values:                 ' +
        #                                     '  | '.join(['{:6.3f}'.format(element) for element
        #                                                  in action_info['q_values']]))
        #         traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
        #                           self.road.road_params['action_info_pos'] + [1, 0]], [0, 0, 0, 0])

        #     if 'aleatoric_std_dev' in action_info:
        #         self.gui_action_info.append('Aleatoric std dev:   ' +
        #                                     '  | '.join(['{:6.3f}'.format(element) for element in
        #                                                  action_info['aleatoric_std_dev']]))
        #         traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
        #                           self.road.road_params['action_info_pos'] + [1, -1*dy]], [0, 0, 0, 0])

        #     if 'epistemic_std_dev' in action_info:
        #         self.gui_action_info.append('Epistemic std dev:  ' +
        #                                     '  | '.join(['{:6.3f}'.format(element) for element in
        #                                                  action_info['epistemic_std_dev']]))
        #         traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
        #                           self.road.road_params['action_info_pos'] + [1, -2*dy]], [0, 0, 0, 0])


    @property
    def nb_actions(self):
        return 3

    @property
    def nb_observations(self):
        return self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles + \
            self.nb_states_per_moped * self.sensor_nb_moped + self.nb_states_per_pedestrian * self.sensor_nb_pedestrians
