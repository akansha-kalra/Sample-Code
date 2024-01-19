__author__ = "akansha_kalra"
import random
import torch
import numpy as np
from Point_Env import Gridworld
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import truncnorm
from numpy.linalg import norm
from utils import toeplitz
# mpl.use('Qt5Agg')
import warnings
warnings.filterwarnings('ignore')
# np.set_printoptions(precision=10)
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
import torch.distributions.multivariate_normal as mvn_distribution

seed=1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


def smooth_traj_initialization(T_planning_horizon,num_input_features,scale=1):
    matrix=toeplitz([1, -2 , 1] + [0]*(T_planning_horizon-1), [1] + [0] * (T_planning_horizon-1))
    matrix[-2:] = 0
    matrix[-1, -2:] = [1, -1]
    covar = matrix.T.dot(matrix)
    inv_cov = torch.from_numpy(scale* np.linalg.inv(covar)).double()
    mu = torch.from_numpy(np.zeros(T_planning_horizon)).double()
    mvn = mvn_distribution.MultivariateNormal(loc=mu, covariance_matrix=inv_cov)

    initial_traj_sample = mvn.rsample((1, T_planning_horizon + 1,))
    initial_traj_sample = initial_traj_sample.reshape(1, T_planning_horizon + 1, 1, num_input_features)
    initial_traj_sample[0,0,:,:]=torch.tensor([1,1])
    return initial_traj_sample

'''Define EBM structure to load the trained model'''
class EBM(nn.Module):
    def __init__(self,num_input_features,dim1,dim2,dim3):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1=nn.Linear(2*num_input_features,dim1,dtype=torch.float64).to(self.device)
        self.fc2=nn.Linear(dim1,dim2,dtype=torch.float64).to(self.device)
        self.fc3=nn.Linear(dim2,dim3,dtype=torch.float64).to(self.device)
        self.fc4=nn.Linear(dim3,1,bias=False,dtype=torch.float64).to(self.device)

    def forward(self,input):
        if input.size(dim=0)==2 and len(input.size())==3:
            input=input.flatten().to(self.device)
        else:
            input=input.reshape(input.size(dim=0),2*input.size(dim=3)).to(self.device)
        h1=F.leaky_relu(self.fc1(input))
        h2=F.leaky_relu(self.fc2(h1))
        h3=F.leaky_relu(self.fc3(h2))
        out=self.fc4(h3)
        return out


class InverseDynamics(nn.Module):
    def __init__(self, action_dim,num_input_features, hidden_dim1):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(2 * num_input_features, hidden_dim1, dtype=torch.float64).to(self.device)
        self.fc2 = nn.Linear(hidden_dim1, action_dim, dtype=torch.float64).to(self.device)

    def forward(self, input):
        if input.size(dim=0) == 2:
            input = input.flatten().to(self.device)
        else:
            input = input.reshape(input.size(dim=0), 2 * input.size(dim=3)).to(self.device)
        h1 = F.relu(self.fc1(input))
        action = self.fc2(h1)
        return action

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_rollouts(initial_traj,K_MC_rollouts,T_planning_horizon,curr_timestep,num_input_features):
    gaussian_distribution_noise = get_truncated_normal(mean=0, sd=1, low=0, upp=1)
    gaussian_noise = torch.from_numpy(gaussian_distribution_noise.rvs((K_MC_rollouts, T_planning_horizon+ 1, num_input_features)))
    gaussian_noise = gaussian_noise.reshape(K_MC_rollouts, T_planning_horizon + 1, 1, num_input_features)
    '''generate rollouts'''
    rollouts = []
    for k in range(K_MC_rollouts):
        if curr_timestep != 0:
            for r in range(curr_timestep+1):
                gaussian_noise[k, r] = 0
        else:
            gaussian_noise[k, 0] = 0
        pertubed_traj = initial_traj + gaussian_noise[k, :]
        # pertubed_traj= torch.clamp(pertubed_traj,min=0,max=2)
        rollouts.append(pertubed_traj)

    return rollouts,gaussian_noise


def get_weights_rollout(env,traj,EBM,l2_include=False):
    power_exp=0
    goal=torch.from_numpy(env.end)
    for l in range(traj.size(dim=1)-1):
        curr_state=traj[0][l]
        ns=traj[0][l+1]
        input_pair=torch.stack((curr_state,ns))
        energy=EBM.forward(input_pair)
        power_exp+=energy
    if l2_include==True:
        '''Since goal state is fixed and NOT a Gaussian distribution'''
        # l2=(traj[0][-1]- goal).pow(2).sum(1)
        final_state_goal_pair=torch.stack((traj[0][-1],goal))
        final_state_goal_energy=EBM.forward(final_state_goal_pair).detach().cpu()
        total_energy=-1*(power_exp.detach().cpu()+final_state_goal_energy)
        # weight=torch.exp(-1*total_energy)
    else:
        # weight=torch.exp(-1* power_exp.detach().cpu())
        total_energy=-1*power_exp.detach().cpu().numpy()

    weight_numpy=np.exp(total_energy)
    # if weight_numpy ==0:
    #     weight = torch.tensor(1e-20)
    # else:
    weight = torch.from_numpy(np.array(weight_numpy))
    return weight


def get_actions(imagined_traj,ID):
    actions=[]
    for l in range(imagined_traj.size(dim=1)-1):
        curr_state = imagined_traj[0][l]
        ns = imagined_traj[0][l + 1]
        input_pair = torch.stack((curr_state, ns))
        curr_action=ID.forward(input_pair)
        actions.append(curr_action.detach().cpu().numpy())
    return actions

def plot_trajs(traj,iter,plotting,color=None):
    # # Clear the entire figure
    # plt.clf()
    if color == None:
        color='black'
    x=[]
    y=[]
    if plotting=="ebm":
        loop_over=traj.size(dim=1)
        traj=traj.squeeze(dim=0)
    elif plotting=="final":
        loop_over=len(traj)
    for l in range(loop_over):
        s=traj[l][0]
        curr_x=s[0]
        curr_y=s[1]
        x.append(curr_x)
        y.append(curr_y)
        plt.text(curr_x, curr_y, str(l))
    # plt.xlim(1, 2)
    # plt.ylim(1, 2)
    plt.scatter(x,y)

    plt.plot(x,y,color)
    if plotting=="final":

        plt.title(f" Optimal traj at iter{iter}")
    elif plotting =="ebm":
        plt.title(f"EBM traj at iter {iter}  without action grounding ")
    plt.show()
    plt.pause(1)
    plt.close()
    # plt.close()

def random_state_space_sampling(T,begin_in_middle):
    random_traj=[]
    x_min=env.start[0,0]

    x_max=env.end[0,0]

    y_min=env.start[0,0]
    y_max=env.end[0,1]

    if begin_in_middle ==True:
        agent_begin=env.set_middle()
        x_middle = env.set_middle()[0, 0]
        y_middle = env.set_middle()[0, 1]
    else:
        agent_begin=env.reset()
    random_traj.append(agent_begin)
    for i in range(1,T+1):
        if begin_in_middle==True:
            random_x_cord=np.random.uniform(low=x_middle,high=x_max)
            random_y_cord=np.random.uniform(low=y_middle,high=y_max)
        else:
            random_x_cord = np.random.uniform(low=x_min, high=x_max)
            random_y_cord = np.random.uniform(low=y_min, high=y_max)

        random_pt=np.array([random_x_cord,random_y_cord]).reshape(1,2)
        random_traj.append(random_pt)

    # random_traj=np.array(random_traj)
    return random_traj

def action_clipping(action):
    l2_norm=norm(action,ord=2)
    if l2_norm<=0.1:
        action_clipped=action
    else:
        action_clipped=0.1*(action/l2_norm)

    return action_clipped


def offline_trained_MPPI(env,EBM_model,ID_model,K_MC_rollouts,T,num_input_features,initial_demonstration,num_iters,printing=False,EBM_L2_included=False):
    optimal_traj=[]
    optimal_actions=[]

    # each state has dim 1xnum_features
    curr_timestep=0
    for iter in range(num_iters):
        '''Initial_demo_torch after iter 0 to be equal to last iter's updated demo'''
        if iter==0:
            initial_demonstration_torch = torch.from_numpy(np.array(initial_demonstration)).reshape(1, T + 1, 1,num_input_features)
        else:
            initial_demonstration_torch=initial_demonstration_updated
            if printing == True:
                print("--------------------------------------" * 20)
                print(f"Input traj to new iter{iter} is {initial_demonstration_torch}")

        if iter == 1:
            print("check")
        rollouts,pertubations=get_rollouts(initial_demonstration_torch,K_MC_rollouts,T,curr_timestep,num_input_features)

        weights = []
        for rollout_index in range(len(rollouts)):
            curr_rollout = rollouts[rollout_index]
            curr_weight = get_weights_rollout(env,curr_rollout, EBM_model, l2_include=EBM_L2_included)
            weights.append(curr_weight)

        # print(weights)


        list_zero_weights=np.where(np.array(weights)==0)[0]
        # if len(list_zero_weights) !=0 and len(list_zero_weights)!= len(rollouts):
        #     min_non_zero_weight = np.min(np.array(weights)[np.nonzero(np.array(weights))])
        #     for i in list_zero_weights:
        #         weights[i]=min_non_zero_weight
        # elif len(list_zero_weights) ==len(rollouts):
        #     for i in list_zero_weights:
        #         weights[i]=torch.tensor(1e-20).double()

        if len(list_zero_weights) != 0  and len(list_zero_weights) == len(rollouts):
            for i in list_zero_weights:
                weights[i] = torch.tensor(1e-20).double()

        weights_torch = torch.stack(weights)
        normalized_weight_dr = torch.sum(weights_torch)
        normalized_weights = weights_torch / normalized_weight_dr



        initial_weighted_delta_traj_noise = []
        for k in range(K_MC_rollouts):
            curr_initial_weighted_delta_traj_noise = normalized_weights[k] * pertubations[k]
            initial_weighted_delta_traj_noise.append(curr_initial_weighted_delta_traj_noise)

        initial_weighted_delta_traj_noise = torch.stack(initial_weighted_delta_traj_noise)
        weighted_delta_traj = (torch.sum(initial_weighted_delta_traj_noise, dim=0)).unsqueeze(dim=0)

        initial_demonstration_updated = initial_demonstration_torch + weighted_delta_traj

        if printing==True:
            print(f" EBM Traj at iter{iter} is {initial_demonstration_updated}")
            plot_trajs(initial_demonstration_updated,iter,plotting='ebm')

        actions_to_simulate = get_actions(initial_demonstration_updated, ID_model)
        action_to_take=actions_to_simulate[curr_timestep]
        real_action_to_take=action_clipping(action_to_take)
        if curr_timestep ==0:
            env.reset()
            curr_state=env.curr_pos
            print(f"curr env state is {curr_state}")
            '''Uncomment for starting at (1,1)'''
            # curr_state = env.set_middle()
            optimal_traj.append(curr_state)
        else:
            env.set_current_state(optimal_traj[curr_timestep])

        ns,r,_,_ =env.step(real_action_to_take)
        optimal_actions.append(real_action_to_take)
        optimal_traj.append(ns)
        if printing==True:
            # print(f"Actions given by ID to simulate at iter{iter} : {actions_to_simulate}")
            print(f"REAL Actions to simulate at iter{iter} : {real_action_to_take}")
            print(f"Traj at iter{iter} is {optimal_traj}")
            plot_trajs(optimal_traj,iter,plotting='final')

        # if iter!=0:
        '''update the input to ebm'''
        for j in range(1,len(optimal_traj)):
            initial_demonstration_updated[0,j,:,:]=torch.from_numpy(optimal_traj[j])
        '''roll the traj one step forward '''
        # initial_demonstration = torch.roll(initial_demonstration_updated, shifts=-1, dims=1)
        curr_timestep+=1
        # plt.close()

    return optimal_traj, optimal_actions




if __name__=="__main__":
    '''Feed this initial_demonstration into ID model to get actions'''
    ID_model=InverseDynamics(action_dim=2,num_input_features=2,hidden_dim1=20)
    ID_model=torch.load('/home/ak/Documents/SelfPOint/Inverse_Dyn_training/saved_models/S1_L2_StepLR_A_53')
    env = Gridworld(seed=seed,radius=0.1, max_time_steps=25)
    # env.seed(seed=seed)
    env.reset()
    done = False
    curr_state = env.curr_pos
    print(f"checking curr_state before mppi loop{curr_state}")

    num_input_features=2

    EBM_model=EBM(num_input_features=num_input_features,dim1=20,dim2=10,dim3=5)
    EBM_model=torch.load('/home/ak/Documents/SelfPOint/EBM_training/Neg_logits/saved_models/NL-2_without_ES_Longer500_500')

    print(EBM_model)
    T = 25

    smooth=False
    '''Right now smooth trajectory only works for T=2 as sampling from any other multivariate distribution means dimensions are off'''
    '''Randomly sample T states- Model free RL/MPC cant use any action grounding '''
    if smooth==True:
        initial_demonstration=smooth_traj_initialization(T,num_input_features)
    else :
        initial_demonstration=random_state_space_sampling(T,begin_in_middle=False)

    K_MC_rollouts=30
    num_iters=25

    print("Initial traj fed to MPPI is ", initial_demonstration)
    optimal_traj,optimal_actions= offline_trained_MPPI(env,EBM_model,ID_model,K_MC_rollouts,T,num_input_features,initial_demonstration,num_iters=num_iters,printing=False,EBM_L2_included=False)

    last_traj_key = num_iters
    print("--------------------------------------" * 20)
    print("Planned traj", optimal_traj)
    plot_trajs(optimal_traj, last_traj_key, plotting='final')
