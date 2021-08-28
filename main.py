import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def csv_statis():
    ##########读取csv中的数据统计状态数########
    df = pd.read_csv('qlearning.csv')
    return df


df = csv_statis()
df.head()

##########统计状态##############
colos = ['pv', 'pload', 'hload', 'pg', 'pt']
state_ = df[colos]
print(state_)
P_pv = df['pv']
P_pv = np.array(P_pv.tolist())
print('P_pv', P_pv)
P_load = df['pload']
P_load = np.array(P_load.tolist())
print('P_load', P_load)
H_load = df['hload']
H_load = np.array(H_load.tolist())
print('H_load', H_load)
P = df['pt']
P = np.array(P.tolist())
print('P', P)
P_g = df['pg']
P_g = np.array(P_g.tolist())
print('P_g', P_g)
V_chp = np.zeros((24,))
print('V_chp', V_chp)
SOC = np.zeros((24,))
p_ess = np.zeros((24,))

import random

C_cs = 0.1
n_chp = 0.5
q_NG = 39.4


def reset():
    ##########环境初始化##############33
    state = {
        'M': 0,
        'SOC': 0,
        'P_pv': 0,
        'P_load': 0,
        'H_load': 0,
        'P': 0,
        'P_g': 0,
        'T': 0
    }

    state['SOC'] = random.randint(0, 2) * 4 + 3
    state['P_pv'] = P_pv[0]
    state['P_load'] = P_load[0]
    state['H_load'] = H_load[0]
    state['P'] = P[0]
    state['P_g'] = P_g[0]

    V_chp[0] = H_load[0] / (1 - n_chp) * q_NG

    P_chp = V_chp[0] * q_NG * n_chp

    state['M'] = - ((P_pv[0] + P_chp) - P_load[0])

    SOC[0] = state['SOC']

    state['T'] = 0

    return str(state)


def step(state, action, T):
    #############env 的 动作部分，按照文档#############
    reward = 0
    next_state = {
        'M': 0,
        'SOC': 0,
        'P_pv': 0,
        'P_load': 0,
        'H_load': 0,
        'P': 0,
        'P_g': 0,
        'T': 0}

    T = T + 1

    SOC[T] = SOC[T - 1] + (action - 1) * 4

    if SOC[T] > 12:
        SOC[T] = 11
        reward = reward - 100
    if SOC[T] < 3:
        SOC[T] = 3
        reward = reward - 100

    V_chp[T] = H_load[T] / (1 - n_chp) * q_NG

    P_chp = V_chp[T] * q_NG * n_chp

    p_ess[T] = (action - 2) * 2

    M = P_pv[T] + P_chp - P_load[T]

    D = P[T] * M

    for t in range(T):
        C = P_g[t] * V_chp[t] + C_cs * p_ess[t]
    if M < 0:
        D = 0
    reward = reward - (C + D)
    ################fixed_state##############33
    next_state['P_pv'] = P_pv[T]
    next_state['P_load'] = P_load[T]
    next_state['H_load'] = H_load[T]
    next_state['P'] = P[T]
    next_state['P_g'] = P_g[T]
    ###################################
    V_chp[T] = H_load[T] / (1 - n_chp) * q_NG

    P_chp = V_chp[T] * q_NG * n_chp

    next_state['M'] = - ((P_pv[T] + P_chp) - P_load[T])

    next_state['SOC'] = SOC[T]

    next_state['T'] = T

    if T == 23:
        done = True
    else:
        done = False
    return reward, str(next_state), T, done


import sys
from collections import defaultdict

LR = 0.1


class Player:
    def __init__(self):
        self.reset()
        return

    def reset(self):
        reset()
        ###self.greedy greedy 策略的探索概论
        self.greedy = 0.1
        return

    def soft_max_behavior_policy(self, Q, state):
        ############波尔兹曼策略############
        ### rt 波尔兹曼策略 退火系数##########
        rt=50
        state_ = eval(state)
        if state_['M'] >= 0:
            print(state_['M'])
            ac_start = 2
            ac_end = 5
        else:
            #print(state_['M'])
            ac_start =0
            ac_end = 3

        t = np.exp(Q[state][ac_start:ac_end]/rt)
        #print(Q[state])
        #print(t)
        a = np.exp(Q[state][ac_start:ac_end]/rt)/np.sum(t, axis=0)
        sample = np.random.uniform(0, 1)
        #print(a)
        if sample <= a[0]:

            act = 0

        elif (sample >= a[0]) and (sample <= a[0] + a[1]):

            act = 1

        elif sample >= (a[0] + a[1]):

            act = 2

        act = ac_start + act

        return act

    def behavior_policy(self, Q, state):
        ##########greedy 策略#############

        state_ = eval(state)
        if state_['M'] >= 0:
            print(state_['M'])
            ac_start = 2
            ac_end = 5
        else:
            #print(state_['M'])
            ac_start =0
            ac_end = 3
        sample = np.random.uniform(0, 1)
        if sample > self.greedy:
            act = np.argmax(Q[state][ac_start:ac_end]) + ac_start
            #print(act)
        else:
            act = np.random.randint(0, 5)
        return act

    def target_policy(self, state, Q):
        value = Q[state]
        return value

    def Qlearning_greedy(self, num_episodes, max_time=100, discount_factor=1.0):
        ###########使用greedy的qlearning#############
        #########初始化Q值############
        self.Q = defaultdict(lambda: np.zeros(5)-2000)
        q_value = []
        xias = []
        for i_episode in range(0, num_episodes + 1):
            if i_episode % 1000 == 0:
                #####每间隔1000个episode进行仿真一次统计该轮reward###########
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
                qq = self.simulate_greedy()
                print(qq)
                q_value.append(qq)
                xias.append(i_episode)
            episode = []
            #####3初始化状态#############3
            t = 0
            state = reset()
            while (1):
                ######根据策略进行决策##########33
                action = self.behavior_policy(self.Q, state)
                # reward,next_state,done = env(action,state)
                #######执行动作#############333
                reward, next_state, next_t, done = step(state=state, action=action, T=t)

                episode.append((state, action, reward))
                # print (next_state)
                # print('argmax',np.argmax(Q[next_state]),'\n')
                ##########Q_learning 更新 q_value###############3
                self.Q[state][action] = self.Q[state][action] + LR * (
                            reward + discount_factor * self.Q[next_state][np.argmax(self.Q[next_state])] -
                            self.Q[state][action])
                state = next_state
                t = next_t
                if done == True:
                    ########经过了24个小时一轮调度完成##############
                    break

        print(len(self.Q))
        # self.Q=Q
        print(q_value)
        ########绘图################3
        #plt.plot(xias, q_value,label='greedy')
        self.q_value_greedy=q_value
        self.xias = xias
        #plt.show
        # print(np.argmax(Q[state]))
        return self.Q

    def Qlearning_soft(self, num_episodes, max_time=100, discount_factor=1.0):
        #########使用波尔兹曼策略的greedy##############
        self.Q = defaultdict(lambda: np.zeros(5)-2000)
        q_value = []
        xias = []
        for i_episode in range(0, num_episodes + 1):
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
                sys.stdout.flush()
                qq = self.simulate_soft()
                q_value.append(qq)
                xias.append(i_episode)
            episode = []
            t = 0
            state = reset()
            while (1):
                action = self.soft_max_behavior_policy(self.Q, state)
                # reward,next_state,done = env(action,state)

                reward, next_state, next_t, done = step(state=state, action=action, T=t)

                episode.append((state, action, reward))
                # print (next_state)
                # print('argmax',np.argmax(Q[next_state]),'\n')
                self.Q[state][action] = self.Q[state][action] + LR * (
                            reward + discount_factor * self.Q[next_state][np.argmax(self.Q[next_state])] -
                            self.Q[state][action])
                state = next_state
                t = next_t
                if done == True:
                    break

        print(len(self.Q))
        # self.Q=Q
        self.q_value_soft=q_value
        self.xias = xias

        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(self.xias, self.q_value_soft, color='blue', linewidth=2, linestyle="-",label='soft')
        axes.plot(self.xias, self.q_value_greedy, color='red', linewidth=2, linestyle="-",label='greedy')
        # axes.set_xlim(x.min(),x.max())
        #axes.set_ylim(min(y2vals.min(), yvals.min()), max(y2vals.max(), yvals.max()))
        #axes.scatter(x_, y_, color="blue", s=200)
        axes.set_xlabel("episode", fontproperties="SimHei", fontsize=14)
        axes.set_ylabel("epi_reward", fontproperties="SimHei", fontsize=14)
        axes.legend()
        fig.savefig('imx.png')
        plt.show()
        # print(np.argmax(Q[state]))
        return self.Q

    def init_Q(self, Q):
        for state in range(7):
            state = state - 3
            Q[state] = [0.0, 0.0, 0.0]
        return Q

    def simulate_greedy(self):
        #######greedy进行仿真##########
        state = reset()
        t = 0
        episode = []
        ep_reward = 0
        while (1):
            action = self.behavior_policy(self.Q, state)
            # reward,next_state,done = env(action,state)

            reward, next_state, next_t, done = step(state=state, action=action, T=t)
            episode.append((action, reward))
            state = next_state
            t = next_t
            ep_reward = ep_reward + reward
            if done == True:
                break
        return ep_reward

    def simulate_soft(self):
        #######波尔兹曼仿真#################
        state = reset()
        t = 0
        episode = []
        ep_reward = 0
        while (1):
            action = self.soft_max_behavior_policy(self.Q, state)
            # reward,next_state,done = env(action,state)

            reward, next_state, next_t, done = step(state=state, action=action, T=t)
            episode.append((action, reward))
            state = next_state
            t = next_t
            ep_reward = ep_reward + reward
            if done == True:
                break
        # print('ep_reward',ep_reward)
        # print('episode',episode)
        return ep_reward

    def simulate_print_soft(self):
        ##########仿真并输出决策##################3
        state = reset()
        t = 0
        episode = []
        ep_reward = 0
        while (1):
            action = self.soft_max_behavior_policy(self.Q, state)
            # reward,next_state,done = env(action,state)

            reward, next_state, next_t, done = step(state=state, action=action, T=t)
            episode.append((action, reward))
            state = next_state
            t = next_t
            ep_reward = ep_reward + reward
            if done == True:
                break
        print('ep_reward', ep_reward)
        print('soft_episode', episode)
        return ep_reward
    def simulate_print_greedy(self):
        ##########仿真并输出决策##################3
        state = reset()
        t = 0
        episode = []
        ep_reward = 0
        while (1):
            action = self.behavior_policy(self.Q, state)
            # reward,next_state,done = env(action,state)

            reward, next_state, next_t, done = step(state=state, action=action, T=t)
            episode.append((action, reward))
            state = next_state
            t = next_t
            ep_reward = ep_reward + reward
            if done == True:
                break
        print('ep_reward', ep_reward)
        print('greedy_episode', episode)
        return ep_reward



player=Player()
_=player.Qlearning_greedy(10000)
player.simulate_print_greedy()
_=player.Qlearning_soft(10000)
player.simulate_print_soft()

