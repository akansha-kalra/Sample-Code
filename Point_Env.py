import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import math
import random
# from random import random
import pygame
import matplotlib.pyplot as plt

class Gridworld(gym.Env):
    def __init__(self,seed,radius,max_time_steps):
        self.seed=seed
        random.seed(self.seed)
        np.random.random(self.seed)
        self.start=np.array([0,0]).reshape(1,2)
        self.end=np.array([2,2]).reshape(1,2)
        self.size=2
        # self.observation_space=spaces.Box(low=self.start, high=self.end, shape=(1,2), dtype=np.uint8)
        self.curr_pos=self.start #default position , update it everytime
        self.counter=0
        # self.max_pts=max_pts
        self.radius=radius
        self.timestep=max_time_steps
        # self.window = None
        # self.clock = None
        # self.window_size = 512
        # self.action_space = spaces.Discrete(8)
        '''
        0 : up , 1: down , 2: right , 3: right , 4: left , 5: 
        '''
        # self.action_to_direction = {0:np.array(0,0.05),1:np.array(0,-0.05),2:np.array(0.05,0),3:np.array()

        # seeding
    # def seed(self, seed):
    #     random.seed(seed)
    #     np.random.random(seed)


    def reset(self):
        # -1 to ensure agent inital position will not be at the end state
        # self.agent_position = np.random.randint((self.size * self.size) - 1)
        self.agent_position =self.start
        self.counter =0
        return self.agent_position
        # return np.array([self.agent_position]).astype(np.uint8)

    def set_current_state(self,curr_s):
        self.curr_pos=curr_s
    def set_middle(self):
        self.middle=np.array([1,1]).reshape(1,2)
        self.set_current_state(self.middle)
        return self.middle

    def points_on_circumference(self,curr_pos):
        '''Randonly sample a point lying on the circumference of a circle centered at curr poss with radius=self.radius
        random.random generates a random no between 0 and 1'''
        random_angle= random.random()*2*math.pi
        random_x=curr_pos[0,0]+ math.cos(random_angle)*self.radius
        random_y=curr_pos[0,1]+ math.sin(random_angle)*self.radius
        random_pt=np.array([random_x,random_y]).reshape(1,2)
        return random_pt

        # ls_pts= [(
        #         curr_pos[0,0] + (math.cos(2 * math.pi / self.max_pts * x) * self.radius),  # x
        #         curr_pos[0,1] + (math.sin(2 * math.pi / self.max_pts * x) * self.radius),  # y
        #     )
        #     for x in range(0, self.max_pts + 1)]
        # return ls_pts

    def pot_ns_validity_check(self,ns):
        x_min=self.start[0,0]
        y_min=self.start[0,1]

        x_max=self.end[0,0]
        y_max=self.end[0,1]

        valid=False
        curr_x=ns[0,0]
        curr_y=ns[0,1]
        if curr_x >= x_min and curr_x <=x_max :
            if curr_y >=y_min and curr_y <=y_max:
                valid=True
            else:
                pass
        else:
            pass
        return valid


    def random_sample_action_space(self,curr_pos):
        # self.curr_pos=curr_pos
        possible_pts=self.points_on_circumference(curr_pos)
        pot_ns=random.choices(possible_pts,k=1)
        pot_ns=np.array(pot_ns).reshape(1,-1)
        is_ns_valid=self.pot_ns_validity_check(pot_ns)
        action = pot_ns - curr_pos
        # if is_ns_valid ==True:
        #     ns=pot_ns
        # else:
        #     ns=curr_pos

        return action

    def step(self, action):

        pot_ns=self.curr_pos+action
        is_ns_valid=self.pot_ns_validity_check(pot_ns)
        if is_ns_valid ==True:
            ns=pot_ns
        else:
            ns=self.curr_pos

        if (ns==self.end).all():
            reward=10
        else:
            '''changed env reward and made sure reward wasnt used anywhere before '''
            dist = np.abs(ns-self.end).sum()
            reward = -1 * dist

        if (ns==self.end).all() or self.counter>=self.timestep-1:
            if (ns==self.end).all():
                print("REACHED GOAL")
            done=True
        else:
            done=False

        info={}
        self.counter += 1
        return ns , reward, done, info

    def check_rendering(self,data):
        plt.xlim([0, 2])
        plt.ylim([0, 2])

        plt.xticks(np.arange(0, 2.1, 0.2))

        plt.yticks(np.arange(0, 2.1, 0.1))
        plt.axis('equal')
        # plt.axis('off')
        for i in range(2):
            # horizontal and vertical background lines
            plt.plot([i, i], [0, 2], linewidth=0.5, color='black')
            plt.plot([0, 2], [i, i], linewidth=0.5, color='black')

        col = []
        offset = np.array([0.1, 0.1])
        for i in range(0, len(data)):
            episode, state = data[i]
            if (state == self.start).all():
                col.append('blue')
            elif (state == self.end).all():
                col.append('green')
            else:
                col.append('black')

        for i in range(len(data)):
            episode, state = data[i]
            plt.scatter(state[0, 0], state[0, 1], c=col[i], s=10, linewidth=0)
            # plt.scatter(state[0,0], state[0,1],s=10,linewidth=0)
            # plt.annotate(str(i + 1), state[0] + offset)
        # plt.tight_layout()
        plt.show()


    def render_grid(self,data,save,episode_no,save_fig_dir):
        # make axis equal and turn them off
        # plt.figure(figsize=(3, 6))
        plt.xlim([0,2])
        plt.ylim([0,2])

        plt.xticks(np.arange(0, 2.1, 0.2))

        plt.yticks(np.arange(0,2.1, 0.1))
        plt.axis('equal')
        # plt.axis('off')
        for i in range(2):
            # horizontal and vertical background lines
            plt.plot([i, i], [0, 2], linewidth=0.5, color='black')
            plt.plot([0, 2], [i, i], linewidth=0.5, color='black')

        col = []
        offset = np.array([0.1, 0.1])

        # for i in range(0, len(data)):
        #     episode,state=data[i]
        #     if (state ==self.start).all():
        #         col.append('blue')
        #     elif (state==self.end).all():
        #         col.append('green')
        #     elif episode==0:
        #         col.append('red')
        #     elif episode==1:
        #         col.append('magenta')
        #     else:
        #         col.append('black')
        for i in range(0, len(data)):
            episode, state = data[i]
            if (state == self.start).all():
                col.append('blue')
            elif (state == self.end).all():
                col.append('green')
            else:
                col.append('black')

        for i in range(len(data)):
            episode, state = data[i]
            plt.scatter(state[0, 0], state[0, 1], c=col[i], s=10, linewidth=0)
            # plt.scatter(state[0,0], state[0,1],s=10,linewidth=0)
            # plt.annotate(str(i + 1), state[0] + offset)
        # plt.tight_layout()
        if save==True:
            plt.savefig(save_fig_dir +f"State visitation map with radius{self.radius} over {episode_no} episodes where each epsiode is {self.timestep} long.png")
        plt.show()

            # class ChopperScape(gym.Env):
#     def __init__(self):
#         super(ChopperScape, self).__init__()
#
#         # Define a 2-D observation space
#         self.observation_shape = (600, 800, 3)
#         self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
#                                             high=np.ones(self.observation_shape),
#                                             dtype=np.float16)
#
#         # Define an action space ranging from 0 to 4
#         self.action_space = spaces.Discrete(6, )
#
#         # Create a canvas to render the environment images upon
#         self.canvas = np.ones(self.observation_shape) * 1
#
#         # Define elements present inside the environment
#         self.elements = []
#
#         # Maximum fuel chopper can take at once
#         self.max_fuel = 1000
#
#         # Permissible area of helicper to be
#         self.y_min = int(self.observation_shape[0] * 0.1)
#         self.x_min = 0
#         self.y_max = int(self.observation_shape[0] * 0.9)
#         self.x_max = self.observation_shape[1]

