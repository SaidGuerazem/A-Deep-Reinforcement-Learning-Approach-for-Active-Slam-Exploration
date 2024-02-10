import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import math

# global_covariance_matrix = np.zeros(3,3)
def amcl_pose_callback(msg):

    covariance_matrix = msg.pose.covariance

    x_covariance = covariance_matrix[0]
    xy_covariance = covariance_matrix[1]
    xtheta_covariance = covariance_matrix[5]
    ytheta_covariance = covariance_matrix[11]
    y_covariance = covariance_matrix[7]
    theta_covariance = covariance_matrix[35]  # Assuming the yaw (theta) is in the last element

    # Creating a 3x3 covariance matrix for (x, y, theta)
    covariance_3x3 = np.array([[x_covariance, xy_covariance, xtheta_covariance],
                               [xy_covariance, y_covariance, ytheta_covariance],
                               [xtheta_covariance, ytheta_covariance, theta_covariance]])
    return covariance_3x3

def D_optimality(Cov_Matrix):
    eigenvalues = np.linalg.eigvals(Cov_Matrix)
    log_eigenvalues = np.array([math.log(eigenvalues[0]), math.log(eigenvalues[1]), math.log(eigenvalues[2])])
    mean_log_eigenvalues = np.mean(log_eigenvalues)
    D_opt = math.exp(mean_log_eigenvalues)
    return D_opt
    

def Additional_rewards_uncertainty():
    cov_Matrix = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)
    D = D_optimality(amcl_pose_callback(cov_Matrix))
    r = math.tanh(0.01/D)
    return r
