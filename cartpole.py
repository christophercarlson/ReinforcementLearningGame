# A reward of +1 is provided for every timestep that the pole remains upright.

# The episode ends when the pole is more than 15 degrees from vertical, or the
# cart moves more than 2.4 units from the center.

import gym
env = gym.make('CartPole-v0')

for i_episode in range(20):
    
    observation = env.reset()
    
    for i in range(100):
        env.render()
         
        print(observation)
       
       
        action = env.action_space.sample()
        
        # step function returns: observation, reward, done, info
        
        observation, reward, done, info = env.step(action)
        #print(reward)
       
       	# done being True indicates the episode has terminated. 
       	# (For example, perhaps the pole tipped too far, or you 
       	# lost your last life.)
       	
        if done:
            print("Episode ends after {} timesteps".format(i+1))
            break