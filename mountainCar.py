import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt


#decrease epsilon a constant amount, so that randomness decreases


env = gym.make("MountainCar-v0", render_mode="rgb_array")
LEARNING_RATE = 0.1
DISCOUNT = 0.95 #how much we value future reward over current reward
EPISODES = 25000
#SHOW_EVERY = 2000 #every 2000 episodes, let us know its alive
SHOW_EVERY = 500 
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) #this is a 20x20 that breaks up continuous data into 20 bins
                                                          #the len is 2 since there are 2 high observation spaces 
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE #width of each bucket
#creating a Q table that is 20x20x3 (3 comes from the # of actions)
q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))
#this is the stepping into environment, make its own method + creating the env  
    

#epsilon = 0.9
epsilon = 1.5
#how likely to perform a random action

#epsilon as .9 gives consistent 6.7-6.8


START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    #want a higher decau value


def updateQTable(q_table, new_discrete_state, discrete_state, action, reward):
    max_future_q = np.max(q_table[new_discrete_state])
    current_q = q_table[discrete_state + (action, )]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) #fancy equation for calculating all q values
    q_table[discrete_state + (action, )] = new_q #update q-table with the new q value
      

def getBestAction(q_table, discrete_state):
    return np.argmax(q_table[discrete_state]) #given current state, which action gets best result


def get_discrete_state(state, env, discrete_os_win_size):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

def runGame(env, epsilon):
    env.reset()
    for episode in range(EPISODES) :
        start_time = time.time()
        if episode % SHOW_EVERY == 0 and episode!=0:
            env = gym.make("MountainCar-v0", render_mode="human")
            
        else:
            env = gym.make("MountainCar-v0", render_mode="rgb_array")
        state, info = env.reset()
        discrete_state = get_discrete_state(state, env, discrete_os_win_size)
        done = False
        while not done:
            if np.random.random() > epsilon:
                action = getBestAction(q_table, discrete_state)
            else:
                action = np.random.randint(0, env.action_space.n)
            new_state, reward, termination, truncation, info = env.step(action)
            #new_state is position and velocity
            done = termination or truncation          
            new_discrete_state = get_discrete_state(new_state, env, discrete_os_win_size)
            if not done:
                updateQTable(q_table, new_discrete_state, discrete_state, action, reward)
            elif new_state[0] >= env.goal_position:
                #print(f"We made it to episode {episode}")
                q_table[discrete_state + (action, )] = 0 #setting reward to 0 (as opposed to -1)

            discrete_state = new_discrete_state
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        if episode % SHOW_EVERY == 0 and episode!=0:
            end_time = time.time()
            episode_time = end_time - start_time
            print(f"Episode {episode} took {episode_time:.2f} seconds to reach the flag.")
   
    # Create a 3D projection       
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # Visualize the array using scatter
    ax.scatter(*np.meshgrid(np.arange(20), np.arange(20), np.arange(3)), c=q_table.ravel(), cmap='viridis')
    plt.show()   
    #z is state 0,1,2
    #dot is q value
    #puple 0, yellow -1    
    env.close()
    


#time each episode to see if each episode gets faster,
runGame(env, epsilon)