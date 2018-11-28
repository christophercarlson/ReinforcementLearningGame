import gym
import matplotlib.pyplot as plt
import numpy as np
import random

# GET HORIZONTAL FEATURE IMPLEMENTED

def main(n_trials=250):
    legalActions = (0, 1)
    features = {"cartPosition":1,"poleAngle":1, "poleVelocity":1}
    weights = [1, 1, 1]
    
    gamma = 0.8
    alpha = 0.1
    epsilon = [1]
    epsilonMin = 0.01
    epsilonDecay = 0.999
    
    testEnv = gym.make('CartPole-v0')
    testEnv = testEnv.env
    testEnv.reset()
    
    numberFails = []
    
    
    def computeAction(observation):
        testValues = {action:computeValueFromAction(action, observation) for action in legalActions}
        return random.choice(legalActions) if (np.random.random() <= epsilon[0]) else max(testValues, key = testValues.get)
    
    
    # What is the qValue of the next state for taking this action
    def computeValueFromAction(action, observation):
        # Set testEnv state to current actual state
        testEnv.state = observation
        # Take hypothetical step in testEnv
        (testObservation, _, done, _) = testEnv.step(action)
        testEnv.reset()
        
        # Make features from future state
        futureFeatures = {"poleAngle":1, "poleVelocity":1}
        getFeatures(futureFeatures, testObservation)
        return -9999 if done else sum([weight * feat for (weight, feat) in zip(weights, futureFeatures.values())])
    
    
    # The sum of all features multiplied by their weights for this state
    def getQValue(features):
        return sum([weight * feat for (weight, feat) in zip(weights, features.values())])
    
    
    # Update the weights
    def update(observation, step_reward, action, features):
        difference = (step_reward + gamma * computeValueFromAction(action, observation)) - getQValue(features)
        for (weight, feature, i) in zip(weights, features.values(), range(len(features))):
            weights[i] = weight + alpha * feature * difference


def decayEpsilon(epsilon):
    if epsilon[0] > epsilonMin:
        epsilon[0] *= epsilonDecay


    def getFeatures(features, observation):
        (cartPosition, cartVelocity, poleAngle, poleVelocity) = observation
        # Pole angle squared
        features["poleAngle"] = poleAngle * poleAngle
        # Rotational velocity squared
        features["poleVelocity"] = poleVelocity * poleVelocity
    # Horizontal position
# features["cartPosition"] = 0.00001 * cartPosition


def episode_reward(weights, epsilon, env=gym.make('CartPole-v0')):
    observation = env.reset()
    total_reward = 0
        
        # try to last 200 steps
        for _ in range(200):
            # Update features based on observation
            getFeatures(features, observation)
            
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
        print("trial: ", trial_nbr, "steps: ", int(reward))
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

