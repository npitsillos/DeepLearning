import gym
import random
import numpy as np
import keras.models as KM
import keras.layers as KL
from keras.optimizers import *

#####################################
#  Network
#####################################
class Brain():

    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count

        self.model = self.build()

    def build(self):

        model = KM.Sequential()
        model.add(KL.Dense(output_dim=64, activation="relu", input_dim=self.state_count))
        model.add(KL.Dense(output_dim=self.action_count, activation="linear"))

        opt = RMSprop(lr=0.00025)
        model.compile(loss="mse", optimizer=opt)
        
        return model

    def train(self, inputs, targets, epochs=1, verbose=0):
        self.model.fit(inputs, targets, batch_size=64, nb_epoch=epochs, verbose=verbose)

    def predict(self, state):
        return self.model.predict(state)

    def predict_one(self, state):
        return self.predict(state.reshape(1, self.state_count)).flatten()


##################################
# Memory
##################################
class Memory():
    
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

###########################################
# Agent
###########################################

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001

class Agent():

    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count

        self.brain = Brain(state_count, action_count)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_count-1)
        else:
            return np.argmax(self.brain.predict_one(state))
    
    def observe(self, sample):
        
        self.memory.add(sample)

        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA*self.steps)

    def replay(self):
        
        # Get samples batch
        batch = self.memory.sample(BATCH_SIZE)
        batch_len = len(batch)
        
        # Denotes terminating state
        no_state = np.zeros(self.state_count, dtype=np.int32)

        # Set up current and next states
        states = np.array([ o[0] for o in batch ], dtype=np.int32)
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ], dtype=np.int32)
        
        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
     
        x = np.zeros((batch_len, self.state_count), dtype=np.int32)
        y = np.zeros((batch_len, self.action_count), dtype=np.int32)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]
            if s_ is None: # finished
                t[a] = r
            else:
                t[a] = r + GAMMA * np.amax(p_[i]) # still to go

        x[i] = s
        y[i] = t

        self.brain.train(x, y)

#######################################
# Environment
#######################################
class Environment():

    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
    
    def run(self, agent):
        s = self.env.reset()
        reward = 0

        while True:
            self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done:
                s_ = None
            
            agent.observe((s, a, r, s_))
            agent.replay()

            s = s_
            reward += r

            if done:
                break

        print("Total reward: ", reward)

env = Environment("CartPole-v0")

state_count = env.env.observation_space.shape[0]
action_count = env.env.action_space.n

agent = Agent(state_count, action_count)

try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-basic.h5")