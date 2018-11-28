import gym
import matplotlib.pyplot as plt
import numpy as np
import random


def main(n_trials=1000):
    legalActions = (0, 1)
    features = [1, 1]
    weights = [1, 1]
    
    gamma = 0.8
    alpha = 0.1
    epsilon = [1]
    epsilonMin = 0.01
    epsilonDecay = 0.999
    
    testEnv = gym.make('CartPole-v0')
    testEnv = testEnv.env
    testEnv.reset()
    
    numberFails = []
    
    
    
    def computeAction(obs):
        testValues = {action:computeValueFromAction(action, obs) for action in legalActions}
        return random.choice(legalActions) if (np.random.random() <= epsilon[0]) else max(testValues, key = testValues.get)
    
    
    # What is the qValue of the next state for taking this action
    def computeValueFromAction(action, obs):
        # Set testEnv state to current actual state
        testEnv.state = obs
        # Take hypothetical step in testEnv
        (t_observation, _, done, _) = testEnv.step(action)
        testEnv.reset()
        
        # Make features from future state
        futureFeatures = [1, 1]
        futureFeatures[0] = (t_observation[2] * t_observation[2])
        futureFeatures[1] = (t_observation[3] * t_observation[3])
        return -9999 if done else sum([weight * feat for (weight, feat) in zip(weights, futureFeatures)])
    
    
    # The sum of all features multiplied by their weights for this state
    def getQValue(features):
        return sum([weight * feat for (weight, feat) in zip(weights, features)])
    
    
    # Update the weights
    def update(observation, step_reward, action, features):
        difference = (step_reward + gamma * computeValueFromAction(action, observation)) - getQValue(features)
        for (weight, feature, i) in zip(weights, features, range(len(features))):
            weights[i] = weight + alpha * feature * difference


def decayEpsilon(epsilon):
    if epsilon[0] > epsilonMin:
        epsilon[0] *= epsilonDecay
    

    def getFeatures(observation):
        # Pole angle squared
        features[0] = (observation[2] * observation[2])
        # Rotational velocity squared
        features[1] = (observation[3] * observation[3])


def episode_reward(weights, epsilon, env=gym.make('CartPole-v0')):
    observation = env.reset()
    total_reward = 0
        
        # try to last 200 steps
        for _ in range(200):
            # Update features based on observation
            getFeatures(observation)
            
            # Find best action
            action = computeAction(observation)
            
            # Step environment based on best action
            (observation, step_reward, done, info) = env.step(action)
            
            total_reward += step_reward
            
            decayEpsilon(epsilon)
            update(observation, step_reward, action, features)
            
            # if _ > 150:
            #     env.render()
            if done:
                return total_reward


def train(trial_nbr):
    reward = episode_reward(weights, epsilon)
    if reward < 200:
        numberFails.append(trial_nbr)
        print(trial_nbr, reward)
        return reward


    def print_plot_results(max_episodes):
        print('\nmax steps: {}; avg steps: {}'. \
              format(max_episodes, round(sum(results)/n_trials, 1)))
              plt.hist(results, bins=50, color='g', density=1, alpha=0.75)
              plt.xlabel('Steps lasted')
              plt.ylabel('Frequency')
              plt.title('Histogram of Pole Weighted Learning')
              plt.show()

# These three lines are the body of main()
# The functions above are nested within main()
results = [train(trial_nbr + 1) for trial_nbr in range(n_trials)]
print('final weights:', [round(w, 3) for w in weights])
print('number of fails: ', len(numberFails))
print_plot_results(max(results))


# Will render if the arg to main() is less than or equal to 5
main()
