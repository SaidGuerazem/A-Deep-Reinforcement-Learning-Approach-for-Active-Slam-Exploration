#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import liveplot
from d3qn import Agent

def concatenate_elements(vector):
     # Assuming the input vector has more than 260 elements
   if len(vector) >= 260:
       concatenated_vector = vector[:100] + vector[-100:]
       return concatenated_vector
   else:
      # Handle the case where the input vector has less than 260 elements
      return "Input vector should have at least 260 elements."

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
    env = gym.make('GazeboCircuitTurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    path = '/tmp/turtle_c_dqn_ep'
    plotter = liveplot.LivePlot(outdir)

    continue_execution = False




    deepQ = Agent(lr=0.005, gamma=0.99, n_actions=3, epsilon=1.0,
                  batch_size=64, input_dims=100)


    env._max_episode_steps = steps # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)
    epochs =10
    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    #start iterating from 'current epoch'.
    for i in range(epochs):
        observation = env.reset()
        print(observation.shape)
        # observation = concatenate_elements(observation)
        cumulated_reward = 0
        done = False
        episode_step = 0

        # run until env returns done
        while not done:
            # env.render()
            print(type(observation))
            # observation = concatenate_elements(observation)
            # print(type(observation))
            # print(observation)
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)
            # newObservation = concatenate_elements(newObservation)
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = newObservation

            if done:
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps "+ "Cumulated rewards"+ str(cumulated_reward) +"  Exploration=" + str(round(explorationRate, 2)))
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%100==0:
                        #save model weights and monitoring data every 100 epochs.
                        deepQ.saveModel(path+str(epoch)+'.h5')
                        env._flush()
                        copy_tree(outdir,path+str(epoch))
                        #save simulation parameters.
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(path+str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

            episode_step += 1

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

        if epoch % 10 == 0:
            plotter.plot(env)

    env.close()

