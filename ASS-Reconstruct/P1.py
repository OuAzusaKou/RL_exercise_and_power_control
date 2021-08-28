from collections import defaultdict
import numpy as np
import sys
from enum import Enum
class ACTION(Enum):
    high='high'
    low='low' 
    
class Winner:
    def __init__(self):
        self.P_H = 0.55
        self.P_L = 0.45
        self.state = 0
        self.done = 0
        self.R = 0
        self.th=0.000001
        self.highcost = -50
        self.lowcost = -10
        self.greedy = 0.1
        self.discount=1.0
        self.init_Q()
        self.init_C()
    def reset_game(self):
        self.state=0
        self.done=0
    def ba_policy(self):
        smp = np.random.uniform(0,1)
        if smp <= 0.5:
            ac = ACTION.high
        else:
            ac = ACTION.low
        return ac
        
    #def ta_policy(self):
    def OffpolicyMCES(self,numepisode):
        self.init_Q()
        self.init_C()
        for i_episode in range(1, numepisode+1):
            if i_episode % 1000 == 0:
                print("\rEpisode {}/{}.".format(i_episode, numepisode), end="")
                sys.stdout.flush()
            
            self.reset_game()
            state=0
            episode = []
            while(1):
                action = self.ba_policy()
                reward,next_state,done = self.env_step(state,action)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state
            G = 0.0
            Weight = 1.0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.discount * G + reward
                self.C[state,action] += Weight
                self.Q[state,action] += (Weight / self.C[state,action]) * (G - self.Q[state,action])
                if state > 1:
                    if action == ACTION.low:
                        Weight = Weight * (2.0)
                    else:
                        Weight=0
                        break
                else:
                    if action == ACTION.high:
                        Weight = Weight*(2.0)
                    else:
                        Weight=0
                        break
            
           
        return self.Q
        
        
    def env_step(self,state,action):
        done=0
        reward=0
        sample = np.random.uniform(0, 1)
        if action == ACTION.high:
            reward = reward + self.highcost
            suc_prob = self.P_H
        else:
            suc_prob = self.P_L
            reward = reward + self.lowcost
        if sample < suc_prob:
            state = state + 1
        else:
            state = state -1
        if state >= 3 :
            reward = reward + 1000
            done = 1
        if state <= (-3):
            done = 1
        return reward,state,done
    def init_Q(self):
        self.Q = defaultdict(lambda:np.zeros(1))
        for state in range(7):
            for act in ACTION:
                    self.Q[state-3,act]=0.0
    def init_C(self):
        
        self.C = defaultdict(lambda:np.zeros(1))
        for state in range(7):
            for act in ACTION:
                    self.C[state-3,act]=0.0
    def reward_expection(self,act,state,V):
        #global ph,ch,pl,cl
        if act == ACTION.high:
            #print('nopro')
            if state == 2:
                expected_reward = self.P_H*(1000+self.highcost+self.discount*V[state+1]) + self.P_L*(self.highcost+self.discount*V[state-1])
            else:
                expected_reward = self.P_H*(self.highcost+self.discount*V[state+1]) + self.P_L*(self.highcost+self.discount*V[state-1])
                #print('exp',expected_reward)
        elif act == ACTION.low:
            if state == 2:
                expected_reward = self.P_L*(1000+self.lowcost+self.discount*V[state+1]) + self.P_H*(self.lowcost+self.discount*V[state-1])
            else:
                expected_reward = self.P_L*(self.lowcost+self.discount*V[state+1]) + self.P_H*(self.lowcost+self.discount*V[state-1])
       
        return expected_reward
    
    def initV(self,V):
        for state in range(7):
            V[state-3] = 0.0
        return V
    def P_E(self,policy):
        V = defaultdict(lambda: np.zeros(0))
        
        
        
        V = self.initV(V)
        
        
        while(1):
            delta = 0
            for state in range (5):
                state = state - 2
                v_buf = V[state]
                #print('policy',policy[state])
                V[state] = self.reward_expection(act=policy[state],state=state,V=V)
                #print('Vs',V[state])
                delta = max(delta,abs(v_buf - V[state]))
                #print('delta',delta)
            if delta < self.th:
                break

        return V,policy
    def cr_pl(self):
        pll = []
        
        for code in range(32):
            buf = defaultdict(lambda: np.zeros(0))
            for k in range (5):
                act = code % 2
                code = code // 2
                state = k - 2
                if act == 1:
                    act = ACTION.high
                else:
                    act= ACTION.low
                buf[state] = act
            pll.append(buf)
        return pll
    winner=Winner()
policylist=winner.cr_pl()
policy_b = defaultdict(lambda: np.zeros(0))
for i in range (32):
    V,policy = winner.P_E(policylist[i])
    policy_b[i]=V[0]
ordered_value=sorted(policy_b.items(),key=lambda x:x[1],reverse=False)
for i in range(len(ordered_value)):
    print('policy_value',format(i),ordered_value[i],'\n')
print('optimal policy \n',policylist[ordered_value[31][0]],'\n','value=',ordered_value[31][1])