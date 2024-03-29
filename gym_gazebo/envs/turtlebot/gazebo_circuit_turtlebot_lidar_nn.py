import gym
import rospy
import roslaunch
import time
import numpy as np
import os
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import subprocess
from geometry_msgs.msg import PoseWithCovarianceStamped
import sys
from sensor_msgs.msg import LaserScan

from gym.utils import seeding
from cov_matrix_retriver import Additional_rewards_uncertainty
def concatenate_elements(vector):
    # Assuming the input vector has more than 260 elements
    if len(vector) >= 260:
        concatenated_vector = vector[:130] + vector[-130:]
        return concatenated_vector
    else:
        # Handle the case where the input vector has less than 260 elements
        return "Input vector should have at least 260 elements."

class GazeboCircuitTurtlebotLidarNnEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboCircuitTurtlebotLidar_v0.launch")
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return data.ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # max_ang_speed = 0.3
        # ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        # vel_cmd = Twist()
        # vel_cmd.linear.x = 0.2
        # vel_cmd.angular.z = ang_vel
            
        # self.vel_pub.publish(vel_cmd)
        
        if action == 0: #FORWARD
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
        elif action == 1: #LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.3
            self.vel_pub.publish(vel_cmd)
        elif action == 2: #RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.3
            self.vel_pub.publish(vel_cmd)
        time.sleep(0.1)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # data.ranges = concatenate_elements(data.ranges)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)

        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            # reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            r = 0
            if action == 0:
                reward = 1 + r
            else:
                reward = -0.05 + r
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            reward = -100

        return np.asarray(state), reward, done, {}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")
        # subprocess.call(['rosnode', 'kill', '/amcl'])
        # new_x_value = 0.0
        # new_y_value = 0.0
        # new_theta_value = 0.0
        # rospy.set_param('/amcl/initial_pose_x', new_x_value)
        # rospy.set_param('/amcl/initial_pose_y', new_y_value)
        # rospy.set_param('/amcl/initial_pose_a', new_theta_value)
        # fullpath = "/opt/ros/melodic/share/turtlebot3_navigation/launch/amcl.launch"
        # port ="11311"
        # ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))
        # subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", port, fullpath])
        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')

        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        # print(data.ranges)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)

        return np.asarray(state)