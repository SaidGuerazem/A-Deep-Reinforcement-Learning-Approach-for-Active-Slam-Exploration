import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# Turtlebot envs
register(
    id='GazeboMazeTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboMazeTurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuitTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuitTurtlebotLidarEnv',
    #timestep_limit = 100,
    #reward_threshold = 1.0,
    #nondeterministic = True,
    # More arguments here
)
register(
    id='GazeboCircuit2TurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuit2TurtlebotLidarEnv',
    # More arguments here
)
register(
    id='GazeboCircuit2TurtlebotLidarNn-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuit2TurtlebotLidarNnEnv',
    max_episode_steps=500,
    # More arguments here
)
###### New added environments in circuit and Maze for DQN app ########
######################################################################
register(
    id='GazeboCircuitTurtlebotLidarNn-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuitTurtlebotLidarNnEnv',
    max_episode_steps=500,
    # More arguments here
)
register(
    id='GazeboMazeTurtlebotLidarNn-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboMazeTurtlebotLidarNnEnv',
    max_episode_steps=500,
    # More arguments here
)
######################################################################
######################################################################
register(
    id='GazeboCircuit2cTurtlebotCameraNnEnv-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboCircuit2cTurtlebotCameraNnEnv',
    max_episode_steps=500,
    
    # More arguments here
)
register(
    id='GazeboRoundTurtlebotLidar-v0',
    entry_point='gym_gazebo.envs.turtlebot:GazeboRoundTurtlebotLidarEnv',
    # More arguments here
)

# Erle-Copter envs
register(
    id='GazeboErleCopterHover-v0',
    entry_point='gym_gazebo.envs.erlecopter:GazeboErleCopterHoverEnv',
)

#Erle-Rover envs
register(
    id='GazeboMazeErleRoverLidar-v0',
    entry_point='gym_gazebo.envs.erlerover:GazeboMazeErleRoverLidarEnv',
)

# Modular SCARA
register(
    id='GazeboModularScara3DOF-v0',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFEnv',
)
register(
    id='GazeboModularScara3DOF-v1',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFv1Env',
)
register(
    id='GazeboModularScara3DOF-v2',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFv2Env',
)
register(
    id='GazeboModularScara3DOF-v3',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFv3Env',
)

register(
    id='GazeboModularScara3DOF-v4',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFv4Env',
)

register(
    id='GazeboModularScara4DOF-v3',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara4DOFv3Env',
)
register(
    id='GazeboModularArticulatedArm4DOF-v1',
    entry_point='gym_gazebo.envs.articulated_arm:GazeboModularArticulatedArm4DOFv1Env',
)

register(
    id='GazeboModularScaraObstacles3DOF-v0',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFObstaclesv0Env',
)
register(
    id='GazeboModularScaraStaticObstacle3DOF-v0',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFStaticObstaclev0Env',
)
register(
    id='GazeboModularScaraStaticObstacle3DOF-v1',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara3DOFStaticObstaclev1Env',
)
register(
    id='GazeboModularScaraArm4And3DOF-v1',
    entry_point='gym_gazebo.envs.modular_scara:GazeboModularScara4And3DOFv1Env',
)
register(
    id='RealModularScara3DOF-v0',
    entry_point='gym_gazebo.envs.modular_scara:RealModularScara3DOFv0Env',
)

# cart pole
register(
    id='GazeboCartPole-v0',
    entry_point='gym_gazebo.envs.gazebo_cartpole:GazeboCartPolev0Env',
)

register(
    id='Box3DOF-v1',
    entry_point='gym_gazebo.envs.modular_scara:Box3DOFv1Env',
)
# ARIACPickv0Env
register(
    id='ARIACPick-v0',
    entry_point='gym_gazebo.envs.ARIAC:ARIACPickv0Env',
)

# MARA
register(
    id='MARASide3DOF-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARASide3DOFv0Env',
)
register(
    id='MARATop3DOF-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARATop3DOFv0Env',
)

register(
    id='MARANoGripper-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARANoGripperv0Env',
)

register(
    id='MARAOrient-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARATopOrientv0Env',
)

register(
    id='MARAVisionOrient-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARATopOrientVisionv0Env',
)
register(
    id='MARAOrientCollision-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARATopOrientCollisionv0Env',
)
register(
    id='MARACollision-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARATopCollisionv0Env',
)
register(
    id='MARAVisionOrientCollision-v0',
    entry_point='gym_gazebo.envs.MARA:GazeboMARATopOrientVisionCollisionv0Env',
)
register(
    id='RealMARA3DoF-v0',
    entry_point='gym_gazebo.envs.MARA:RealModularMara3DOFv0Env',
)
