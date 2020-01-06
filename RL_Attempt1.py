import numpy as np
import tensorflow as tf
'''
#Implement REINFORCE

#declare variables for states, actions, rewards, weights, policy        
#initiate loop of # of cycles
    
    #as episode runs, store log probabilites of policy and rewards at each step
        #calculate discounted reward at each step
        #calucalate the new policy with the policy gradient
    
    #repeat again for new trial with new policy
'''

def __init__(self):
    self.gamma = .95 #discount factor
        
    self.state_memory = [] #initial state
    self.action_memory = [] #initial action
    self.reward_memory = [] #initial reward
    
    
cycles = 10 #probably do more

#Agent
def learn(self):
    for i in range(cycles):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory) #these should be the new variables keeping track of memory

        G = np.zeros_like(reward_memory) #policy in array of 0's same as reward_memory
        
        #need to store log probabilities of policy and rewards of each step
        '''
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        '''
        
        for t in range(len(reward_memory)): #so loop runs as long as there are numbers in reward memory loop 
            G_sum = 0
            discount = 1
            #for each reward add to the policy the reward after taking into account the discount
            for j in range(t, len(reward_memory)):
                G_sum += reward_memory[j] * discount #this should optimize the policy using gradient ascent over time
                discount *= self.gamma #decreases as reward is farther from last reward, from same state
          
            G[t] = G_sum #afterwards make equal to policy for that episode
        
        # these are used to optimize the policy
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G-mean) / std # lowers variance 
        
        #resets agents memory so loop can run again without leak
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
            
    #then we repeat for number of cycles with the goal of reaching an optimized policy
    #should have a way to store as well to remember once optimal policy is reached
        
    #ton of stuff probably missing, try to have it solve cart problem so ik what is
        

