import gym
import numpy as np
#from gym import wrappers

'''SO THIS CODE RUNS THE CART PROBLEM SUCCESSFULLY KEEEPING IT UP till 200 moves making it successful. 
Understand it since it is a successful implementaion of REINFORCEMENT LEARNING'''

env = gym.make('CartPole-v0') #this is the environment

bestLenght = 0 #initial length
episode_lenghts = [] #empty array for episode lenghts, filled later

best_weights = np.zeros(4) #makes array size 4 of 0's

for i in range(100): #runs for 100 games and we find average later
    new_weights = np.random.uniform(-1.0, 1.0, 4) #(low, high, size) returns 'size' amount of numbers

    lenght = []
    for j in range(100):
        observation = env.reset()
        #print(observation, ' -  this is the observation') #observation was 4 random numbers generated differently each loop same episode
        done = False
        cnt = 0
        
        while not done:
#            env.render()
            cnt += 1
            action = 1 if np.dot(observation, new_weights) > 0 else 0  #np.dot returns the dot product of observation * new_wieghts

            observation, reward, done, _ = env.step(action) #so this command takes an 'action' each step and returns 4 parameters: observation, reward, done, info
            # so itll give an array/object, floating data, bool, diagnostic used for debugging which is why its left as _
            #https://medium.com/@ashish_fagna/understanding-openai-gym-25c79c06eccb
            #this like is the: action put into environment, environment returns reward and observation, and repeat!!!!!

            if done: #this is returned true from env.step if parameters met
                break

        lenght.append(cnt) #a.append(b) adds b to end of a array, so adds count to end of lenght

    average_lenght = float(sum(lenght) / len(lenght)) #this runs after all the trials done, finds total average

    if average_lenght > bestLenght:
        bestLenght = average_lenght
        best_weights = new_weights
    episode_lenghts.append(average_lenght) #keeps adding best average to episode lenght array

    if i % 10 == 0: #very smart method prints best lenght every 10 games
        print(bestLenght, '  < best lenght ', i)
#reset params
done = False
cnt = 0
#env = wrappers.Monitor(env, 'MovieFiles2', force = True)
observation = env.reset()

#so this final loop runs it one last time, but with the best weights, and those optimized weights (after 100 cycles) allow it to solve the problem
while not done:
    cnt += 1
    action = 1 if np.dot(observation, best_weights) > 0 else 0
    
    observation, reward, done, _ = env.step(action) #should return done when solved
    
    if done:
        break


print(cnt, 'moves')


