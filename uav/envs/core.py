import numpy as np
import seaborn as sns
import math

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # Set internal state set for uav or mbs
      
        self.hasFile = []
        self.fileRequest = None
        self.x = None
        self.y = None

class UserState(EntityState):
    def __init__(self):
        # Set internal state set for user
        self.associate = []
        self.hasFile = []
        self.x = None
        self.y = None
        self.fileRequest = None


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

        # entity can move / be pushed
        self.movable = False

        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()


# properties of agent entities
class Agent(Entity):
    def __init__(self, isMBS):
        super(Agent, self).__init__()
        # agent are MBS
        if isMBS == True:
            self.isUAV = False
            self.isMBS = True
        else:
            self.isUAV = True
            self.isMBS = False

        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
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
        self.state = UserState()
        self.movable = False
        self.mbs_associate = None
        self.user_associate = None
        self.file_size = file_size
        self.zipf_parameter = zipf_parameter
        self.file_request = np.random.zipf(1 / zipf_parameter, file_size)


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
                print(f'MBS action: {agent.action}, shape: {agent.action.shape}')
                self.mbs_apply_agent_association(agent.action, self.agents)
            else:
                print(f'UAV action: {agent.action}')
                if len(agent.action) == 4:
                    # Set position, Set cache, set power
                    self.uav_apply_cache(agent.action[0], agent)
                    self.uav_apply_power(agent.action[1], agent)
                    self.uav_apply_trajectory(agent.action[2], agent.action[3], agent)
                else:
                    NotImplementedError

        for user in self.users:
            self.update_user_state(user)
            
        for agent in self.agents:
            self.calculateReward(agent)    


    def calculateReward(self, state, action):
        # Receiving state and action as list
        epsilon = 0.2
        # step 1. Get extr reward (Action에 대해서 state를 모두 바꾸었을 때, reward를 계산)
        extr_reward = calcExtrReward()

        # step 2. Get intr reward
        intr_reward = calcIIntrReward()

        return extr_reward + epsilon * intr_reward

    def calcIIntrReward(self):
        NotImplementedError

    def calcExtrReward(self):
        # step 1. action에 대한 status 변경
        for agent in self.agents:
            if agent.isUAV == False:
                print("Process for MBS")
            else:
                print("Process for UAV")

        # step 2. 변경된 status로 reward 계산
        L = 100000000  # very large number
        Delay = 0
        for agent in self.agents:
            for user in self.num_user:
                Delay += GetDelay(self, agent, user, agent.isUAV)

        return L - Delay

    def GetDelay(self, node, user, isMbs):
        if isMbs == True:  # for MBS
            for file in self.num_files:
                delay = (
                    self.x(user, file) * self.z(user, node) * self.T_down(node, user)
                )
        else:  # for User
            for file in self.num_files:
                delay = (
                    self.x(user, file)
                    * self.z(user, node)
                    * {
                        self.T_down(node, user)
                        + (1 - self.y(node, file)) * self.T_back(node, user)
                    }
                )

        return delay

    def Calc_T_down(self):
        return S / R_T_down(self, i, u)

    def Calc_T_back(self):
        return S / R_T_back(self, b, m, u)

    def R_T_down(self, i, u):
        numfile = 0
        for file in self.num_files:
            numfile += x(u, file) * z(u, i)

        upper = numfile * W
        lower = 0
        for user in self.num_users:
            for file in self.num_files:
                lower += x(u, file) * z(u, i)

        return upper / lower * math.log2(1 + r(i, u))

    def R_T_back(self, b, m, u):
        left = math.log2(1 + r(b, m))
        upper, lower = 0
        for file in self.num_files:
            upper += x(u, file) * z(u, m) * (1 - y(m, file))
        upper *= B

        for user in self.num_users:
            for file in self.num_files:
                for node in self.num_agents:
                    lower += x(user, file) * z(user, file) * (1 - y(node, file))

        return left * upper / lower

    def r(self, i, u, type):
        if type == TYPE_MBS_USER:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, h_MbsUser(self, i, u) / 10))

        elif type == TYPE_UAV_USER:  # follows UAV Power
            lower = 0
            for uavIdx in self.num_uavs:
                if uavIdx == i:
                    continue
                lower += P(uavIdx) * math.pow(10, -h_UavUser(i, u) / 10)

            res = P(i) * math.pow(10, -h_UavUser(i, u) / 10) / (NOISE_POWER * lower)

        elif type == TYPE_MBS_UAV:
            res = MBS_POWER / (NOISE_POWER * math.pow(10, h_MbsUav(self, i, u) / 10))

        else:
            res = 0

        return res

    # Calculate pathloss
    def h_UavUser(self, m, u):
        return PLos(m, u) * hLos(m, u) + (1 - PLos(m, u)) * hNLos(m, u)

    def h_MbsUav(self, b, m):
        return PLos(b, m) * hLos(b, m) + (1 - PLos(b, m)) * hNLos(b, m)

    def h_MbsUser(self, b, u):
        return 15.3 + 37.6 * math.log10(d(self, b, u))

    def PLos(self, m, u):
        return 1 / (1 + c_1 * math.exp(-c_2 * (theta(m, u) - c_1)))

    def hLos(self, m, u):
        return 20 * math.log(4 * math.pi * d(self, m, u) / v_c) + X_Los

    def hNLos(self, m, u):
        return 20 * math.log(4 * math.pi * d(self, m, u) / v_c) + X_NLos

    # Calculate Distance


    def mbs_apply_agent_association(self, action_set, agent_list):
        association = action_set.sample
        print(f'[mbs_apply_agent_association] {agent}, {agent_set}, {association}')
        if len(action_set) == self.num_agents:
            # for agent1 in self.agents:
            #     for agent2 in self.        
            NotImplementedError
        else:
            NotImplementedError
        
    def uav_apply_cache(self, action_cache, agent):
        print(f'[uav_apply_cache] {agent}, {action_cache}')
        cache = action_cache.sample
        NotImplementedError
        
    def uav_apply_power(self, action_power, agent):
        data = action_power.sample()
        print(f'[uav_apply_power] {agent}, {data}')
        for i in len(agent.association):
            agent.power[i] = data / len(agent.association)
        
    def uav_apply_trajectory(self, action_dist, action_angle,agent):
        print(f'[uav_apply_trajectory] {agent}, {action_dist}, {action_angle}')
        dist = action_dist.sample()
        angle = action_dist.sample()  # 0~360
        print(f'traj_action : {action_dist.sample()}, {action_angle.sample()}')

        agent.state.x =  agent.state.x + dist * math.cos(angle)
        agent.state.y =  agent.state.y + dist * math.sin(angle)        
                
    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                # force = mass * a * action + n
                p_force[i] = (
                    agent.mass * agent.accel if agent.accel is not None else agent.mass
                ) * agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    def update_user_state(self, user):
        print(f'[update_user_state] {user}, {user.state}')
        # Check new cache file request
        # Calculate remaining download size
        

    # get collision forces for any contact between two entities
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        if self.cache_dists:
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == "H":
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (
            ent_pos[prll_dim] < wall.endpoints[0] - entity.size
            or ent_pos[prll_dim] > wall.endpoints[1] + entity.size
        ):
            return None  # entity is beyond endpoints of wall
        elif (
            ent_pos[prll_dim] < wall.endpoints[0]
            or ent_pos[prll_dim] > wall.endpoints[1]
        ):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force
