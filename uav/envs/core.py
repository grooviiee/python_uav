import numpy as np
import seaborn as sns
import math

# Rate calculation type
TYPE_MBS_USER = 0
TYPE_UAV_USER = 1
TYPE_MBS_UAV = 2

S = 10 * 1024 * 1024  # 10 Mbits
B = 20 * 10 ^ 6
W = 10 * 10 ^ 6
MBS_POWER = 2  # Watt
SPEED_OF_LIGHT = 3 * 10 ^ 8
CARRIER_FREQEUENCY = 2 * 10 ^ 9
QUOTA_UAV = 4
QUOTA_MBS = 20
PATHLOSS_EXP = 2
NOISE_POWER = -100  # dB/Hz

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
        self.power_alloc = []
        
        
        # script behavior to execute
        self.action_callback = None

        # zoe 20200420
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
        self.agents = []
        self.users = []
        self.walls = []
        
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
        print(f'[WORLD_STEP] USER state update done: {len(self.users)}')
        
        for agent in self.agents:
            agent.reward = self.calculateReward(agent)    


    def calculateReward(self, agent):
        epsilon = 0.2
        print(f"[CALC_REWARD] Start Calculating rewards. epsilon: {epsilon}")
        # step 1. Get extr reward (Action에 대해서 state를 모두 바꾸었을 때, reward를 계산)
        extr_reward = self.calcExtrReward(agent)
        # step 2. Get intr reward
        intr_reward = self.calcIntrReward(agent)
        reward = extr_reward + epsilon * intr_reward
        print(f"[CALC_REWARD] reward: {reward}, {extr_reward}, {epsilon}, {intr_reward}")
        return reward
    

    def calcIntrReward(self, agent):
        print(f"[CALC_REWARD] Skip Calculating Intr. rewards.")
        return 0 #Temp

    def calcExtrReward(self, agent):
        # step 2. 변경된 status로 reward 계산
        L = 100000000  # very large number
        Delay = 0
        for agent in self.agents:
            agent.reward = 0
            print(f"[CALC_REWARD] Get AGENT({agent.agent_id})-USER{agent.state.association}.")
            
            for user_id in agent.state.association:
                user = self.users[user_id]
                agent.reward += self.getDelay(agent, user, user.user_id, agent.isUAV)

            Delay += agent.reward

        return L - Delay

    def getDelay(self, agent, user, user_id, isUAV):
        #     if isMbs == True:
        #         for file in range(self.num_files):
        #             delay = (self.x(user, file) * self.z(user, node) * self.T_down(node, user))
        #     else:
        #         for file in range(self.num_files):
        #             delay = (self.x(user, file) * self.z(user, node) * {self.T_down(node, user) + (1 - self.y(node, file)) * self.T_back(node, user)})
        delay = 0
        if isUAV == False:
            print(f"[CALC_REWARD] GetDelay {agent.agent_id} || {user.state.file_request}")
            delay = self.Calc_T_down(agent, user)
        else:
            print(f"[CALC_REWARD] HasFile {agent.state.has_file} || File_request {user.state.file_request}")
            if user.state.file_request in agent.state.has_file:
                delay = self.Calc_T_down(agent, user) + self.Calc_T_back(agent, user)

        return delay


    def Calc_T_down(self, agent, user):
        return S / self.R_T_down(agent, user)

    def Calc_T_back(self, agent, user):
        return S / self.R_T_back(b, agent, user)

    def R_T_down(self, mbs, user): # i : mbs , u: user
        numfile = 0
        for file in range(self.num_files):
            numfile += self.x(user, file) * self.z(user, mbs)

        upper = numfile * W
        lower = 0
        for user_idx, user in enumerate(self.users):
            for file in range(self.num_files):
                lower += self.x(user, file) * self.z(user, mbs)

        if lower == 0:
            return 0.0001

        r_t_down = upper / lower * math.log2(1 + self.r(mbs, user))
        
        print(f"[CALC_REWARD] R_T_down between {mbs}, {user}: {r_t_down}")
        return r_t_down

    def R_T_back(self, b, m, u):
        left = math.log2(1 + self.r(b, m))
        upper, lower = 0
        for file in self.num_files:
            upper += self.x(u, file) * self.z(u, m) * (1 - self.y(m, file))
        upper *= B

        for user in self.num_users:
            for file in self.num_files:
                for node in self.num_agents:
                    lower += self.x(user, file) * self.z(user, node) * (1 - self.y(node, file))

        return left * upper / lower

    def r(self, i, u, type):
        if type == TYPE_MBS_USER:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, self.h_MbsUser(self, i, u) / 10))

        elif type == TYPE_UAV_USER:  # follows UAV Power
            lower = 0
            for uavIdx in self.num_uavs:
                if uavIdx == i:
                    continue
                lower += self.GetPower(uavIdx, i) * math.pow(10, -self.h_UavUser(i, u) / 10)

            res = self.GetPower(uavIdx, i) * math.pow(10, -self.h_UavUser(i, u) / 10) / (NOISE_POWER * lower)

        elif type == TYPE_MBS_UAV:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, self.h_MbsUav(self, i, u) / 10))

        else:
            res = 0

        return res

    # Calculate pathloss
    def getPower(self, uav_id, user_id):
        # getUserIdxFromAssociation
        agent = self.agents[uav_id]
        user_idx = None
        for idx, value in enumerate(agent.state.association):
            if value == user_id:
                user_idx = idx
                break
            
        return agent.state.power[user_idx]
    
    def h_UavUser(self, m, u):
        return self.PLos(m, u) * self.hLos(m, u) + (1 - self.PLos(m, u)) * self.hNLos(m, u)

    def h_MbsUav(self, b, m):
        return self.PLos(b, m) * self.hLos(b, m) + (1 - self.PLos(b, m)) * self.hNLos(b, m)

    def h_MbsUser(self, b, u):
        return 15.3 + 37.6 * math.log10(self.d(b, u))

    def PLos(self, m, u):
        return 1 / (1 + c_1 * math.exp(-c_2 * (theta(m, u) - c_1)))

    def hLos(self, m, u):
        return 20 * math.log(4 * math.pi * self.d(m, u) / v_c) + X_Los

    def hNLos(self, m, u):
        return 20 * math.log(4 * math.pi * self.d(m, u) / v_c) + X_NLos

    def x(self, user, file):
        if user.state.file_request == file:
            return 1
        else:
            return 0

    def y(self, node, file):
        if file in node.state.has_file:
            return 1
        else:
            return 0
        
    def z(self, node, user):
        if user in node.state.association:
            return True
        else:
            return False

    def d(self, m, u):
        agent = self.agents[m]
        user = self.users[u]
        x = agent.state.x - user.state.x
        y = agent.state.y - user.state.y
        
        return math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    # Calculate Distance


    def mbs_apply_agent_association(self, action_set):
        association = action_set # [nodes][users]
        tmp_association = [[1 for j in range(self.num_agents)] for i in range(self.num_users)]
        
        #init 
        for i, node in enumerate(self.agents):
            node.state.association = []
            for j, user in enumerate(self.users):
                user.state.association = []

        #set
        for i, node in enumerate(self.agents):
            for j, user in enumerate(self.users):
                if tmp_association[j][i]:
                    print(f'[mbs_apply_agent_association] Set agent: {i}, user: {j} TRUE')
                    node.state.association.append(j)
                    user.state.association.append(i)

        
    def uav_apply_cache(self, action_cache, agent):
        #print(f'[uav_apply_cache] agent_id: {agent}, cache: {action_cache}')
        agent.state.has_file = action_cache
        NotImplementedError
        
    def uav_apply_power(self, action_power, agent):
        #print(f'[uav_apply_power] {agent}, {action_power}')
        for i in range(len(agent.association)):
            agent.power[i] = action_power / len(agent.association)
            agent.state.power[i] = agent.power[i]
        
    def uav_apply_trajectory(self, action_dist, action_angle, agent):
        #print(f'[uav_apply_trajectory] {agent}, prev: {agent.state.x}, {agent.state.y}')
        agent.state.x =  agent.state.x + action_dist * math.cos(action_angle)
        agent.state.y =  agent.state.y + action_dist * math.sin(action_angle)     
        #print(f'[uav_apply_trajectory] {agent}, curr: {agent.state.x}, {agent.state.y}')   

    def update_user_state(self, user):
        #print(f'[update_user_state] {user}')
        # Check new cache file request
        NotImplemented
