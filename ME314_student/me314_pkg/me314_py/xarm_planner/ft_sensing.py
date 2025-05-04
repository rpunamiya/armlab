#!/usr/bin/env python3

import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import tf2_ros
from rclpy.duration import Duration

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper
from geometry_msgs.msg import WrenchStamped
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time


class ForceSensing(Node):
    def __init__(self):
        super().__init__('Force_Sensing_Node')

        # Initialize variables to store the latest force/torque data
        self.FT_force_x = 0.0
        self.FT_force_y = 0.0
        self.FT_force_z = 0.0
        self.FT_torque_x = 0.0
        self.FT_torque_y = 0.0
        self.FT_torque_z = 0.0
        
        # Create a subscription to the force/torque sensor topic
        self.ft_ext_state_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ft_ext_state_cb, 10)

        # Create a subscription to the ball world coordinates topic
        # self.ball_world_sub = self.create_subscription(Pose, '/ball_world_coords', self.ball_world_coords_cb, 10)
        
        # Create a timer that calls move_log_ft every 1.0 seconds (1 Hz)
        self.timer = self.create_timer(1.0, self.move_log_ft)
        
        # Log a message to indicate the node has started
        self.get_logger().info("FT Monitor started - logging at 1.0 Hz")
        
        # Replace the direct publishers with the command queue publisher
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        
        # Subscribe to current arm pose and gripper position for status tracking (optional)
        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        
        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

        self.init_arm_pose = None
        self.contact_pose = None
        self.final_pose = None
        self.ball_world_coords = None

        self.strain = []
        self.stress = []

    # def ball_world_coords_cb(self, msg: Pose):
    #     ball_x = msg.position.x
    #     ball_y = msg.position.y
    #     ball_z = msg.position.z
    #     ball_qx = msg.orientation.x
    #     ball_qy = msg.orientation.y
    #     ball_qz = msg.orientation.z
    #     ball_qw = msg.orientation.w
    #     self.ball_world_coords = [ball_x, ball_y, ball_z, ball_qx, ball_qy, ball_qz, ball_qw]

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def publish_pose(self, pose_array: list):
        """
        Publishes a pose command to the command queue using an array format.
        pose_array format: [x, y, z, qx, qy, qz, qw]
        """
        # Create a CommandQueue message containing a single pose command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the pose command
        wrapper = CommandWrapper()
        wrapper.command_type = "pose"
        
        # Populate the pose_command with the values from the pose_array
        wrapper.pose_command.x = pose_array[0]
        wrapper.pose_command.y = pose_array[1]
        wrapper.pose_command.z = pose_array[2]
        wrapper.pose_command.qx = pose_array[3]
        wrapper.pose_command.qy = pose_array[4]
        wrapper.pose_command.qz = pose_array[5]
        wrapper.pose_command.qw = pose_array[6]
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published Pose to command queue:\n"
                               f"  position=({pose_array[0]}, {pose_array[1]}, {pose_array[2]})\n"
                               f"  orientation=({pose_array[3]}, {pose_array[4]}, "
                               f"{pose_array[5]}, {pose_array[6]})")

    def publish_gripper_position(self, gripper_pos: float):
        """
        Publishes a gripper command to the command queue.
        For example:
          0.0 is "fully open"
          1.0 is "closed"
        """
        # Create a CommandQueue message containing a single gripper command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the gripper command
        wrapper = CommandWrapper()
        wrapper.command_type = "gripper"
        wrapper.gripper_command.gripper_position = gripper_pos
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published gripper command to queue: {gripper_pos:.2f}")
    
    def ft_ext_state_cb(self, msg: WrenchStamped):
        """
        Callback function that runs whenever a new force/torque message is received.
        
        This function extracts the force and torque data from the message
        and stores it for later use.
        
        Args:
            msg (WrenchStamped): The force/torque sensor message
        """
        if self.init_arm_pose is None:
            self.init_arm_pose = self.current_arm_pose
        
        # Extract force components from the message
        self.FT_force_x = msg.wrench.force.x
        self.FT_force_y = msg.wrench.force.y
        self.FT_force_z = msg.wrench.force.z
        
        # Extract torque components from the message
        self.FT_torque_x = msg.wrench.torque.x
        self.FT_torque_y = msg.wrench.torque.y
        self.FT_torque_z = msg.wrench.torque.z

    def move_log_ft(self):
        """
        Timer callback function that logs force/torque data at a fixed rate (1 Hz).
        
        This function:
        1. Calculates the magnitude of force and torque vectors
        2. Logs the individual components and magnitudes
        """
        # Calculate force magnitude using the Euclidean norm (square root of sum of squares)
        force_magnitude = math.sqrt(self.FT_force_x**2 + self.FT_force_y**2 + self.FT_force_z**2)
        
        # Calculate torque magnitude
        # torque_magnitude = math.sqrt(self.FT_torque_x**2 + self.FT_torque_y**2 + self.FT_torque_z**2)
        
        # Log the force data (components and magnitude)
        self.get_logger().info(f"Force: [{self.FT_force_x:.2f}, {self.FT_force_y:.2f}, {self.FT_force_z:.2f}] N")
        self.get_logger().info(f"Force magnitude: {force_magnitude:.2f} N")

        if self.contact_pose is not None:
            displacement = self.contact_pose.position.z - self.current_arm_pose.position.z
            self.get_logger().info(f"Displacement: {displacement:.2f} m")

            width = 5.1 / 1000
            length = 32.2 / 1000
            area = width * length
            stress = force_magnitude / area
            strain = displacement / (2 * 0.0508)  # original ball length of 0.0508m
            self.strain.append(strain)
            self.stress.append(stress)
        
        # Log the torque data (components and magnitude)
        # self.get_logger().info(f"Torque: [{self.FT_torque_x:.2f}, {self.FT_torque_y:.2f}, {self.FT_torque_z:.2f}] Nm")
        # self.get_logger().info(f"Torque magnitude: {torque_magnitude:.2f} Nm")

        if self.contact_pose is None:
            if force_magnitude > 0.4:
                self.contact_pose = self.current_arm_pose
                self.get_logger().info("Contact detected.")
            else:
                self.get_logger().info("No contact detected.")
                # move the robot down
                self.get_logger().info("Current pose: "
                                       f"[{self.current_arm_pose.position.x:.2f}, "
                                       f"{self.current_arm_pose.position.y:.2f}, "
                                       f"{self.current_arm_pose.position.z:.2f}]")
                self.publish_pose([self.current_arm_pose.position.x,
                                   self.current_arm_pose.position.y,
                                   self.current_arm_pose.position.z - 0.005,
                                   self.current_arm_pose.orientation.x,
                                   self.current_arm_pose.orientation.y,
                                   self.current_arm_pose.orientation.z,
                                   self.current_arm_pose.orientation.w])
        else:
            if force_magnitude < 0.5:
                self.get_logger().info("Force is below 0.5N, moving robot 0.01m down.")
                self.publish_pose([self.current_arm_pose.position.x,
                                   self.current_arm_pose.position.y,
                                   self.current_arm_pose.position.z - 0.001,
                                   self.current_arm_pose.orientation.x,
                                   self.current_arm_pose.orientation.y,
                                   self.current_arm_pose.orientation.z,
                                   self.current_arm_pose.orientation.w])
            elif force_magnitude < 1:
                self.get_logger().info("Force is between 0.5N and 1N, moving robot 0.001m down.")
                self.publish_pose([self.current_arm_pose.position.x,
                                   self.current_arm_pose.position.y,
                                   self.current_arm_pose.position.z - 0.0001,
                                   self.current_arm_pose.orientation.x,
                                   self.current_arm_pose.orientation.y,
                                   self.current_arm_pose.orientation.z,
                                   self.current_arm_pose.orientation.w])
            else:
                self.get_logger().info("Force is above 1N, stopped the robot.")
                if self.final_pose is None:
                    self.final_pose = self.current_arm_pose
                    self.get_logger().info("Final pose set.")

def main(args=None):
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # <─ key line
    # rclpy.init(args=args)
    node = ForceSensing()

    # Let's first close the gripper (0.0 to 1.0, where 0.0 is fully open and 1.0 is fully closed)
    node.get_logger().info("Closing gripper...")
    node.publish_gripper_position(1.0)

    node.ball_world_coords = [0.28570734, -0.04599861, 0.05104559, 0.0, 0.0, 0.0, 1.0]  # Hard coded ball world coordinates

    # Move to 10cm above the ball
    node.get_logger().info("Moving to 1cm above the ball...")
    node.publish_pose([node.ball_world_coords[0], node.ball_world_coords[1],
                       node.ball_world_coords[2] + 0.01,
                       node.ball_world_coords[3], node.ball_world_coords[4],
                       node.ball_world_coords[5], node.ball_world_coords[6]])
    while node.current_arm_pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    while abs(node.current_arm_pose.position.z - node.ball_world_coords[2]) > 0.02:
        rclpy.spin_once(node, timeout_sec=1.0)  # <<< process ROS messages
        if node.contact_pose is not None:
            break

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down node.")

        node.get_logger().info("All actions done. Shutting down.")
        node.publish_pose([node.init_arm_pose.position.x, node.init_arm_pose.position.y,
                        node.init_arm_pose.position.z, node.init_arm_pose.orientation.x,
                        node.init_arm_pose.orientation.y, node.init_arm_pose.orientation.z,
                        node.init_arm_pose.orientation.w])
        node.publish_gripper_position(0.0)  # Open the gripper
        node.get_logger().info("Gripper opened and robot moved to initial pose.")

        # Save the strain and stress data to a csv file
        data = {
            'strain': node.strain,
            'stress': node.stress
        }
        df = pd.DataFrame(data)
        df.to_csv('strain_stress_data.csv', index=False)
        node.get_logger().info("Strain and stress data saved to strain_stress_data.csv.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    matplotlib.use('Agg')          # guarantees no GUI errors

    df = pd.read_csv('strain_stress_data.csv')

    # Scatter plot
    plt.scatter(df['strain'], df['stress'], color='blue', label='Data')

    # Line of best fit
    X = df['strain'].values.reshape(-1, 1)
    y = df['stress'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Plot the line
    plt.plot(df['strain'], y_pred, color='red', label='Best Fit Line')

    # R^2 score
    r2 = r2_score(y, y_pred)
    plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top')

    # Labels and grid
    plt.xlabel('Strain')
    plt.ylabel('Stress (Pa)')
    plt.title('Stress–Strain Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('stress_strain_curve.png', dpi=150)
    print("Curve saved to stress_strain_curve.png")

if __name__ == '__main__':
    main()