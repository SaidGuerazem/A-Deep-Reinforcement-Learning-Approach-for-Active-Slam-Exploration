import random
import tensorflow as tf
import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense, Activation, LeakyReLU, Dropout
from keras.models import load_model
from keras.regularizers import l2
import os
import memory
tf.enable_eager_execution()
class DuelingDeepQNetwork(tf.keras.Model):
    def __init__(self, n_actions, fcl1_dim, fcl2_dim):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = Dense(fcl1_dim, activation='relu')
        self.dense2 = Dense(fcl2_dim, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)
    def call(self, state):
        # print("State before transfo",state.shape)
        state = tf.expand_dims(state, axis=0) # Add a batch dimension
        # print("State after transfo", state)
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        # print("Advantage after transfo", A)
        Q = (V + (A - tf.math.reduce_mean(A, axis=2, keep_dims=True)))
        # print("Q function after transfo", Q)
        return Q
    
    def advantage(self, state):
        state = tf.expand_dims(state, axis=0)  # Add a batch dimension
        state = tf.cast(state, dtype=tf.float32)
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
        return A
    
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_shape), dtype= np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype= np.float32)

        self.action_memory = np.zeros((self.mem_size), dtype= np.int32)
        self.reward_memory = np.zeros((self.mem_size), dtype= np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype= np.bool)
    
    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, new_states, dones
    
class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,input_dims, epsilon_dec =1e-3, eps_end = 0.01,mem_size = 10000, fcl1_dim = 24, fcl2_dim = 24, replace = 100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DuelingDeepQNetwork(n_actions, fcl1_dim, fcl2_dim)
        self.q_next = DuelingDeepQNetwork(n_actions, fcl1_dim, fcl2_dim)

        opt = tf.keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-06)
        self.q_eval.compile(optimizer= opt, loss='mean_squared_error')
        self.q_next.compile(optimizer=opt, loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            Ad = self.q_eval.advantage(state)
            QQ = Ad.numpy()[0]
            action = np.argmax(QQ,axis=1)
            # print("shape of the chosen action", action.shape)
        return action
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
                
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
                
        q_pred = self.q_eval(states)
        q_next = self.q_next(new_states)
        q_next = q_next.numpy()
            
        q_target = q_pred.numpy()
        # print("Q eval shape", self.q_eval(new_states).numpy().shape)
        QQ = self.q_eval(new_states).numpy()[0]
        max_actions = np.argmax(QQ,axis=1).reshape(self.batch_size,1)
        # print(max_actions.shape)
        # print(max_actions)
        # print(max_actions.shape)
        # print(tf.math.argmax(self.q_eval.predict(new_states),axis=1))
        # print(tf.math.argmax(self.q_eval.predict(new_states),axis=0))
            # Improvement on the solution
        for idx, terminal in enumerate(dones):
            q_target[:,idx, int(actions[idx])] = rewards[idx] + self.gamma * q_next[:,idx, int(max_actions[idx])] * (1 - int(dones[idx]))
        
        q_target = q_target[0]
        # print(q_target.shape)
        # print(states.shape)
        self.q_eval.fit(states, q_target, batch_size = self.batch_size, epochs=1, verbose = 0)  
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter +=1
    def save_Model(self, path):
        self.q_eval.save_weights(path)