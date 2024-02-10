#!/usr/bin/env python

import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import liveplot
from d3qn import Agent
import numpy as np
import matplotlib.pyplot as plt
from gym.wrappers import Monitor
def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)


if __name__ == '__main__':

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    path = '/tmp/turtle_c_d3qn_ep'
    plotter = liveplot.LivePlot(outdir)
    ###############################################
    continue_execution = False
    #fill this if continue_execution=True
    resume_epoch = '101' # change to epoch to continue from
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = resume_path
    params_json  = resume_path + '.json'
    steps = 500
    env._max_episode_steps = steps # env returns done after _max_episode_steps
    env = Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()
    ###############################################

    agent = Agent(lr=0.00025, gamma=0.99, n_actions=3, epsilon=1.0,
                  batch_size=64, input_dims=100)
    n_episodes = 500
    scores =[]
    eps_history = []
    episodes = []


    for i in range(n_episodes):
        episode_step = 0
        done = False
        score = 0
        stepCounter=0
        cumulated_reward = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            # print("THis is the fucking action chosen:", action)
            newobservation, reward, done, info = env.step(action)
            score +=reward
            agent.store_transition(observation, action, reward,
                                    newobservation, done)
            observation = newobservation
            agent.learn()
            stepCounter += 1
            if done:
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if last100Filled:
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(i) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(score) + "   Eps=" + str(round(agent.epsilon, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (i)%100==0:
                        agent.save_Model(path)
                        env._flush()
                        # copy_tree(outdir,path+str(i))
                        # #save simulation parameters.
                        # parameter_keys = ['epochs','explorationRate','minibatch_size','learningRate','gamma','Rewards']
                        # parameter_values = [i, agent.epsilon, agent.batch_size, agent.lr, agent.gamma, scores]
                        # parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        # with open(path+str(i)+'.json', 'w') as outfile:
                        #     json.dump(parameter_dictionary, outfile)

        eps_history.append(agent.epsilon)
        scores.append(score)
        episodes.append(i)

        # avg_score = np.mean(score[-100:])
        print('Episode', i , 'score', score, "Steps", stepCounter)
        if i % 10 == 0:
            plotter.plot(env)
    env.close()
            

