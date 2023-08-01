import numpy as np
import math
import logging

# Rate calculation type
TYPE_MBS_USER = 0
TYPE_UAV_USER = 1
TYPE_MBS_UAV = 2

S = 10 * 1024 * 1024  # 10 Mbits
B = 20 * 10 ^ 6
W = 10 * 10 ^ 6
H = 10
MBS_POWER = 2  # Watt
SPEED_OF_LIGHT = 3 * 10 ^ 8
CARRIER_FREQEUENCY = 2 * 10 ^ 9
QUOTA_UAV = 4
QUOTA_MBS = 20
PATHLOSS_EXP = 2
NOISE_POWER = -100  # dB/Hz
X_Los = 6 #dB
X_NLos = 20 #dB
c_1 = 11.9
c_2 = 0.13
v_c = 3 * 10^8 # speed of light

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # User and Agent in common
        self.x = None
        self.y = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self, cache_size):
        super(AgentState, self).__init__()
        # Set internal state set for uav or mbs
        # in common
        self.association = []
        # for UAV
        self.has_file = []
        self.cache_size = cache_size
        self.file_request = []
        self.power = []
        # for MBS

class UserState(EntityState):
    def __init__(self):
        # Set internal state set for user
        self.association = []
        self.file_request = 0


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.moveX = None
        # communication action
        self.moveY = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ""

        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()


# properties of agent entities
class Agent(Entity):
    def __init__(self, isMBS, cache_capa):
        super(Agent, self).__init__()
        # agent are MBS
        if isMBS == True:
            self.isUAV = False
            self.isMBS = True
        else:
            self.isUAV = True
            self.isMBS = False

        self.agent_id = None
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState(cache_capa)
        # action: physical action u & communication action c
        self.action = Action()
        self.association = []
        self.mbs_associate = None
        self.user_associate = None        
        
        # script behavior to execute
        self.action_callback = None

        print(f"Create agent as isMBS: {isMBS}")


class User(Entity):
    def __init__(self, file_size, num_file, zipf_parameter):
        self.user_id = None
        self.state = UserState()
        self.movable = False
        self.mbs_associate = None
        self.user_associate = None
        self.file_size = file_size
        self.zipf_parameter = zipf_parameter
        #self.state.file_request = np.random.zipf(1 / zipf_parameter, file_size)


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []        # {mbs + uav} dtype: list
        self.users = []         # dtype: list
        
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

        self.num_mbs = 0
        self.num_uavs = 0
        self.world_length = 25
        self.world_step = 0
        self.num_agents = 0
        self.num_users = 0
        self.map_size = 0
        self.num_files = 0
        self.file_size = 0
        self.zipf_parameter = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.users

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]


    # user color
    def assign_user_colors(self):
        for user in self.users:
            user.color = np.array([0.25, 0.50, 0.25])

    def assign_agent_colors(self):
        for agent in self.agents:
            if agent.isUAV == False:
                agent.color = np.array([0.25, 0.25, 0.50])
            else:
                agent.color = np.array([0.50, 0.25, 0.25])

    # update state of the world
    def world_take_step(self):
        print(f"[WORLD_STEP] Take a step in core")
        self.world_step += 1

        for agent in self.agents:
            if agent.isUAV == False:
                # Set association
                print(f'[WORLD_STEP] MBS action: {len(agent.action)}')
                self.mbs_apply_agent_association(agent.action)
            else:
                print(f'[WORLD_STEP] UAV action: {len(agent.action)}')
                # Set position, Set cache, set power
                self.uav_apply_cache(agent.action[0][0], agent)
                self.uav_apply_power(agent.action[0][1], agent)
                self.uav_apply_trajectory(agent.action[0][2], agent.action[0][3], agent)

        for user in self.users:    
            self.update_user_state(user)
        
        for agent in self.agents:
            agent.reward = self.calculateReward(agent)    


    def calculateReward(self, agent):
        epsilon = 0.2
        if self.log_level >= 4:
            print(f"[CALC_REWARD] Start Calculating rewards. epsilon: {epsilon}")
            
        # step 1. Get extr reward (Action에 대해서 state를 모두 바꾸었을 때, reward를 계산)
        extr_reward = self.calcExtrReward(agent)
        # step 2. Get intr reward
        intr_reward = self.calcIntrReward(agent)
        reward = extr_reward + epsilon * intr_reward
        if self.log_level >= 4:
            print(f"[CALC_REWARD] reward: {reward}, {extr_reward}, {epsilon}, {intr_reward}")
        return reward
    

    def calcIntrReward(self, agent):
        if self.log_level >= 4:
            print(f"[CALC_REWARD] Skip Calculating Intr. rewards.")
        return 0 #Temp

    def calcExtrReward(self, agent):
        # step 2. 변경된 status로 reward 계산
        L = 100000000  # very large number
        Delay = 0
        for agent in self.agents:
            agent.reward = 0
            if self.log_level >= 4:
                print(f"[CALC_REWARD] Get AGENT({agent.agent_id})-USER{agent.state.association}.")
            
            for user_id in agent.state.association:
                user = self.users[user_id]
                agent.reward += self.getDelay(agent, user, self.agents[0], agent.isUAV)

            Delay += agent.reward

        return L - Delay

    def getDelay(self, agent, user, mbs, isUAV):
        #     if isMbs == True:
        #         for file in range(self.num_files):
        #             delay = (self.x(user, file) * self.z(user, node) * self.T_down(node, user))
        #     else:
        #         for file in range(self.num_files):
        #             delay = (self.x(user, file) * self.z(user, node) * {self.T_down(node, user) + (1 - self.y(node, file)) * self.T_back(node, user)})
        delay = 0
        if isUAV == False:
            if self.log_level >= 4:
                print(f"[CALC_REWARD] GetDelay {agent.agent_id} || {user.state.file_request}")
            delay = self.Calc_T_down(agent, user, TYPE_MBS_USER)
        else:
            if self.log_level >= 4:
                print(f"[CALC_REWARD] HasFile {agent.state.has_file}, {type(agent.state.has_file)} || File_request {user.state.file_request}, {type(user.state.file_request)}")
            if np.isin(agent.state.has_file, user.state.file_request):
                # Only consider UAV-User
                delay = self.Calc_T_down(agent, user, TYPE_UAV_USER)
            else:
                # Consider backhaul network also
                delay = self.Calc_T_down(agent, user, TYPE_UAV_USER) + self.Calc_T_back(mbs, agent)

        return delay


    # it could be MBS-User or UAV-User
    def Calc_T_down(self, agent, user, type):
        return S / self.R_T_down(agent, user, type)

    # MBS - UAV
    def Calc_T_back(self, mbs, uav):
        backhaul_rate = self.R_T_back(mbs, uav)
        if backhaul_rate == 0:
            backhaul_rate = 0.00000001

        return S / backhaul_rate

    # Tx to User.. i : mbs or uav , u: user, x: file req, y: has file, z: asso
    def R_T_down(self, mbs, user, type): 
        numfile = 1
        upper = numfile * W
        lower = 1
        # for user_idx, user in enumerate(self.users):
        #     for file in range(self.num_files):
        #         lower += self.x(user, file) * self.z(user, mbs)

        r_t_down = upper / lower * math.log2(1 + self.calc_rate(mbs, user, type))
        if self.log_level >= 4:
            print(f"[CALC_REWARD] R_T_down between {mbs.agent_id}, {user.user_id}: {r_t_down}")
        return r_t_down

    def R_T_back(self, mbs, uav):
        left = math.log2(1 + self.calc_rate_MBS_UAV(mbs, uav))
        # upper = 0
        # lower = 0
        # for file in self.num_files:
        #     upper += self.x(u, file) * self.z(u, m) * (1 - self.y(m, file))
        upper = 1
        upper *= B

        lower = 1
        # for user in self.num_users:
        #     for file in self.num_files:
        #         for node in self.num_agents:
        #             lower += self.x(user, file) * self.z(user, node) * (1 - self.y(node, file))
        
        if self.log_level >= 4:
            print(f"left: {left}, upper: {upper}, lower: {lower}")
        return left * upper / lower

    # calculate rate
    def calc_rate_MBS_USER(self, src, dst):
        res = MBS_POWER / (NOISE_POWER * math.pow(10, self.h_MbsUser(src, dst) / 10))
        return res
    
    def calc_rate_UAV_USER(self, src, dst):
        lower = self.GetPower(src, dst) * math.pow(10, -self.h_UavUser(src, dst) / 10)
        res = self.GetPower(src, dst) * math.pow(10, -self.h_UavUser(src, dst) / 10) / (NOISE_POWER * lower)
        return res

    def calc_rate_MBS_UAV(self, src, dst):
        res = MBS_POWER / (NOISE_POWER * math.pow(10, self.h_MbsUav(src, dst) / 10))
        if self.log_level >= 4:
            print(f"MBS_POWER: {MBS_POWER}, NOISE_POWER: {NOISE_POWER}, res: {res}")
        return res
         
    def calc_rate(self, src, dst, type):
        if type == TYPE_MBS_USER:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, self.h_MbsUser(src, dst) / 10))

        elif type == TYPE_UAV_USER:  # follows UAV Power
            # lower = 0
            # for uavIdx in self.num_uavs:
            #     if uavIdx == i:
            #         continue
            #     lower += self.GetPower(uavIdx, i) * math.pow(10, -self.h_UavUser(src, dst) / 10)
            lower = self.GetPower(src, dst) * math.pow(10, -self.h_UavUser(src, dst) / 10)
            res = self.GetPower(src, dst) * math.pow(10, -self.h_UavUser(src, dst) / 10) / (NOISE_POWER * lower)

        elif type == TYPE_MBS_UAV:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, self.h_MbsUav(src, dst) / 10))
            print(f"MBS_POWER: {MBS_POWER}, NOISE_POWER: {NOISE_POWER}, res: {res}")
        else:
            res = 0

        return res

    # Calculate pathloss
    def GetPower(self, uav, user):
        # getUserIdxFromAssociation
        user_idx = user.user_id
        for idx, value in enumerate(uav.state.association):
            if value == user_idx:
                user_id = idx
                break
            
        return uav.state.power[user_id]
    
    def h_UavUser(self, m, u):
        return self.PLos(m, u) * self.hLos(m, u) + (1 - self.PLos(m, u)) * self.hNLos(m, u)

    def h_MbsUav(self, b, m):
        return self.PLos(b, m) * self.hLos(b, m) + (1 - self.PLos(b, m)) * self.hNLos(b, m)

    def h_MbsUser(self, b, u):
        return 15.3 + 37.6 * math.log10(self.d(b, u))

    def PLos(self, m, u):
        return 1 / (1 + c_1 * math.exp(-c_2 * (self.theta(m, u) - c_1)))

    def hLos(self, m, u):
        return 20 * math.log(4 * math.pi * self.d(m, u) / v_c) + X_Los

    def hNLos(self, m, u):
        return 20 * math.log(4 * math.pi * self.d(m, u) / v_c) + X_NLos

    # user requests file
    def x(self, user, file):
        if user.state.file_request == file:
            return 1
        else:
            return 0

    # uav has file
    def y(self, uav, file):
        if file in uav.state.has_file:
            return 1
        else:
            return 0
    
    # node associated with user
    def z(self, node, user):
        if user in node.state.association:
            return True
        else:
            return False

    def theta(self, uav, user):
        return 180/math.pi*math.asin(H/self.d(uav,user))

    # Calculate Distance
    def d(self, uav, user):
        x = uav.state.x - user.state.x
        y = uav.state.y - user.state.y        
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

    def mbs_apply_agent_association(self, action_set):
        association = action_set # [nodes][users]
        tmp_association = [[1 for j in range(self.num_agents)] for i in range(self.num_users)]
        
        # init association
        for i, node in enumerate(self.agents):
            node.state.association = []
            for j, user in enumerate(self.users):
                user.state.association = []

        # set uav and user states following action
        for i, node in enumerate(self.agents):
            for j, user in enumerate(self.users):
                if tmp_association[j][i]:
                    print(f'[mbs_apply_agent_association] Set agent: {i}, user: {j} TRUE')
                    node.state.association.append(j)
                    user.state.association.append(i)

        
    def uav_apply_cache(self, action_cache, agent):
        print(f'[uav_apply_cache] agent_id: {agent}, cache: {action_cache}')
        if action_cache.size > agent.state.cache_size:
            print(f"[uav_apply_cache] agent_id: {agent}, action_space overs cache_size: ({action_cache}/{agent.state.cache_size})")
            agent.state.cache_size = []
            for _, file in enumerate(action_cache):
                agent.state.cache_size.append(file)
        else: # add all files to UAV
            agent.state.has_file = action_cache
        
    def uav_apply_power(self, action_power, agent):
        print(f'[uav_apply_power] {agent}, {action_power}')
        agent.power = action_power
        for user_id in range(self.num_users):
            if user_id in agent.state.association:
                power = agent.power / len(agent.state.association)
                agent.state.power.append(power)
            else:
                agent.state.power.append(0)
        
    def uav_apply_trajectory(self, action_dist, action_angle, agent):
        prev_x = agent.state.x
        prev_y = agent.state.y
        agent.state.x =  agent.state.x + action_dist * math.cos(action_angle)
        agent.state.y =  agent.state.y + action_dist * math.sin(action_angle)
     
        print(f'[uav_apply_trajectory] {agent}, prev: {prev_x}, {prev_y}, curr: {agent.state.x}, {agent.state.y}')   

    def update_user_state(self, user):
        #print(f'[update_user_state] {user}')
        # Check new cache file request
        NotImplemented
