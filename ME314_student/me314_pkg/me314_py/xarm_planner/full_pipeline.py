#!/usr/bin/env python3

import rclpy
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
import sys
sys.path.append("/home/rohan/xarm_ros2_ws/src/armlab/Grounded-SAM-2/")
from sam_hsv import VisualDetector
import time
import pandas as pd

from rclpy.signals import SignalHandlerOptions

# Import the command queue message types from the reference code
from geometry_msgs.msg import WrenchStamped
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper

REAL = True

class Pipeline(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        
        # Replace the direct publishers with the command queue publisher
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)

        # Create publisher for ball world coordinates
        # self.ball_world_coords_pub = self.create_publisher(Pose, '/ball_world_coords', 10)
        
        # Subscribe to current arm pose and gripper position for status tracking (optional)
        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        self.init_arm_pose = None

        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)
        
        self.bridge = CvBridge()

        if REAL:
            color_sub = '/camera/realsense2_camera_node/color/image_raw'
            depth_sub = '/camera/realsense2_camera_node/aligned_depth_to_color/image_raw'
        else:
            color_sub = '/color/image_raw'
            depth_sub = '/aligned_depth_to_color/image_raw'

        self.subscription = self.create_subscription(
            Image,
            color_sub,
            self.camera_listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.subscription_depth = self.create_subscription(
            Image,
            depth_sub,
            self.depth_listener_callback,
            10)
        self.subscription_depth  # prevent unused variable warning

        # Intrinsics for RGB and Depth cameras
        if REAL:
            self.rgb_K = (605.763671875, 606.1971435546875, 324.188720703125, 248.70957946777344)
        else:
            self.rgb_K = (640.5098266601562, 640.5098266601562, 640.0, 360.0)

        self.ball_center_coordinates = None
        self.ball_depth = None

        self.buffer_length = Duration(seconds=5, nanoseconds=0)
        self.tf_buffer = tf2_ros.Buffer(cache_time=self.buffer_length)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.found = False
        self.gotDepth = False

        self.poses_around_ball = []
        self.poses_to_ball = []
        self.visualDetector = VisualDetector()
        self.masked_image = None
        self.mask = None
        self.masked_image_surround = None
        self.mask_surround = None

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
        # self.timer = self.create_timer(1.0, self.move_log_ft)
        
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
        self.stress_stop_counter = 0

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg

    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def publish_pose(self, pose_array: list):
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        wrapper = CommandWrapper()
        wrapper.command_type = "pose"
        
        wrapper.pose_command.x = pose_array[0]
        wrapper.pose_command.y = pose_array[1]
        wrapper.pose_command.z = pose_array[2]
        wrapper.pose_command.qx = pose_array[3]
        wrapper.pose_command.qy = pose_array[4]
        wrapper.pose_command.qz = pose_array[5]
        wrapper.pose_command.qw = pose_array[6]
        
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published Pose to command queue:\n"
                               f"  position=({pose_array[0]}, {pose_array[1]}, {pose_array[2]})\n"
                               f"  orientation=({pose_array[3]}, {pose_array[4]}, "
                               f"{pose_array[5]}, {pose_array[6]})")

    def publish_gripper_position(self, gripper_pos: float):
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        wrapper = CommandWrapper()
        wrapper.command_type = "gripper"
        wrapper.gripper_command.gripper_position = gripper_pos
        
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published gripper command to queue: {gripper_pos:.2f}")

    def camera_listener_callback(self, msg):

        if msg is None:
            self.get_logger().error("Received an empty image message!")
            return
        
        if self.ball_center_coordinates is None:
            if self.current_arm_pose is not None:
                pose = self.current_arm_pose
                if self.init_arm_pose is None:
                    self.init_arm_pose = pose

                new_pose = [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z + 0.01,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w
                ]
                
                self.publish_pose(new_pose)
                self.get_logger().info("Raising camera to look for objects...")
                
                cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, "rgb8")

                self.get_logger().info("Getting mask...")

                try:
                    masked_image, mask, ball_center = self.visualDetector.get_mask(cv_ColorImage)
                except ValueError as e:
                    self.get_logger().warn(f"Could not unpack values from get_mask: {e}")
                    return  # optionally skip further processing

                
                if ball_center != (None, None):
                    self.get_logger().info(f"Found ball at: {ball_center}")
                    self.ball_center_coordinates = ball_center
                    self.masked_image = masked_image
                    self.mask = mask
        else:
            cv_ColorImage = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            masked_image, mask, ball_center = self.visualDetector.get_mask(cv_ColorImage)
            if ball_center != (None, None):
                self.masked_image_surround = masked_image
                self.mask_surround = mask
            else:
                self.get_logger().info("Ball not found in the image.")

    def depth_listener_callback(self, msg):
        if self.ball_center_coordinates is None:
            return
        else:
            aligned_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            bx, by = self.ball_center_coordinates
            self.ball_depth = aligned_depth[by, bx]

    def pixel_to_camera_frame(self, pixel_coords, depth_m):
        fx, fy, cx, cy = self.rgb_K
        u, v = pixel_coords
        X = (u - cx) * depth_m / fx
        Y = (v - cy) * depth_m / fy
        Z = depth_m
        return (X, Y, Z)

    def camera_to_base_tf(self, camera_coords, frame_name: str):
        try:
            if_loop = self.tf_buffer.can_transform('world', frame_name, rclpy.time.Time(), timeout=Duration(seconds=2.0))
            while not if_loop:
                self.get_logger().info("Waiting for transform...")
                rclpy.spin_once(self)
                if_loop = self.tf_buffer.can_transform('world', frame_name, rclpy.time.Time(), timeout=Duration(seconds=2.0))
            if self.tf_buffer.can_transform('world',
                                            frame_name,
                                            rclpy.time.Time()):
                transform_camera_to_base = self.tf_buffer.lookup_transform(
                    'world',
                    frame_name,
                    rclpy.time.Time())

                tf_geom = transform_camera_to_base.transform

                trans = np.array([tf_geom.translation.x,
                                  tf_geom.translation.y,
                                  tf_geom.translation.z], dtype=float)
                rot = np.array([tf_geom.rotation.x,
                                tf_geom.rotation.y,
                                tf_geom.rotation.z,
                                tf_geom.rotation.w], dtype=float)

                transform_mat = self.create_transformation_matrix(rot, trans)
                camera_coords_homogenous = np.array([[camera_coords[0]],
                                                     [camera_coords[1]],
                                                     [camera_coords[2]],
                                                     [1]])
                base_coords = transform_mat @ camera_coords_homogenous
                return base_coords
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to convert camera->base transform: {str(e)}")
            return None

    def create_transformation_matrix(self, quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = translation
        return matrix

    def look_at_quaternion(self, ee_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Returns (qx,qy,qz,qw) so that the tool's +Z axis looks at target_pos.
        """
        z_axis = target_pos - ee_pos
        norm = np.linalg.norm(z_axis)
        if norm < 1e-6:
            raise ValueError("EE position too close to target; cannot orient.")
        z_axis /= norm

        # Pick an arbitrary X axis not (almost) parallel to z_axis
        x_axis_guess = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(x_axis_guess, z_axis)) > 0.95:
            x_axis_guess = np.array([0.0, 1.0, 0.0])

        y_axis = np.cross(z_axis, x_axis_guess)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)          # re-orthogonalise
        x_axis /= np.linalg.norm(x_axis)

        rot_mx = np.column_stack((x_axis, y_axis, z_axis))   # 3×3
        quat_xyzw = R.from_matrix(rot_mx).as_quat()          # (x,y,z,w)
        return quat_xyzw
    
    def is_pose_close(self, current_pose, target_pose, position_threshold=0.01, orientation_threshold=0.01):
        """
        Check if the current pose is close to the target pose within given thresholds.
        """
        current_position = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
        target_position = np.array(target_pose[:3])
        position_distance = np.linalg.norm(current_position - target_position)
        current_orientation = np.array([current_pose.orientation.x, current_pose.orientation.y,
                                        current_pose.orientation.z, current_pose.orientation.w])
        target_orientation = np.array(target_pose[3:])
        orientation_distance = np.linalg.norm(current_orientation - target_orientation)
        return position_distance < position_threshold and orientation_distance < orientation_threshold

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

def main(args=None):
    rclpy.init(args=args)
    node = Pipeline()

    # Find the ball and get its depth
    while not node.found or not node.gotDepth:
        rclpy.spin_once(node)
        if node.ball_center_coordinates is not None:
            node.found = True
        if node.ball_depth is not None:
            node.gotDepth = True

    # Get the average HSV values and reflectance of the ball
    average_hsv = []
    average_hsv_new = node.visualDetector.average_hsv_masked(node.masked_image)
    node.get_logger().info(f"Average HSV: {average_hsv_new}")
    average_hsv.append(average_hsv_new)
    refl_mean, refl_var = [], []
    refl_mean_new, refl_var_new = node.visualDetector.extract_reflectance(node.masked_image, node.mask)
    node.get_logger().info(f"Reflectance Mean: {refl_mean_new}, Variance: {refl_var_new}")
    refl_mean.append(refl_mean_new)
    refl_var.append(refl_var_new)

    # Open the gripper
    node.get_logger().info("Opening gripper...")
    node.publish_gripper_position(0.0)

    # Find the world coordinates of the ball
    camera_coords_ball = node.pixel_to_camera_frame(node.ball_center_coordinates, node.ball_depth/1000.0)
    node.get_logger().info(f"Ball camera coords: {camera_coords_ball}")
    world_coords_ball = node.camera_to_base_tf(camera_coords_ball, 'camera_color_optical_frame')
    node.get_logger().info(f"Ball world coords: {world_coords_ball}")

    # Calculate five observation poses (orbit in the y-z plane, 10 cm radius)
    radius_m     = 0.10
    angles_deg   = [60, 75, 90, 105, 120]

    target_ball = np.array([world_coords_ball[0, 0] - 0.066,
                            world_coords_ball[1, 0],
                            world_coords_ball[2, 0]])
    node.get_logger().info(f"Target ball: {target_ball}")

    node.get_logger().info("Generating observation poses…")
    for k, ang in enumerate(angles_deg, 1):
        theta = np.deg2rad(ang)
        dy    =  radius_m * np.cos(theta)
        dz    =  radius_m * np.sin(theta)

        ee_p  = np.array([world_coords_ball[0, 0] - 0.066,
                        world_coords_ball[1, 0] + dy,
                        world_coords_ball[2, 0] + dz])

        try:
            qx, qy, qz, qw = node.look_at_quaternion(ee_p, target_ball)
        except ValueError as exc:
            node.get_logger().error(f"Pose {k}: {exc}")
            continue

        pose_k = [ee_p[0], ee_p[1], ee_p[2], qx, qy, qz, qw]
        node.poses_around_ball.append(pose_k)
        node.get_logger().info(f"Obs pose {k}: {pose_k}")

    # Publish the observation poses, and get the HSV and reflectance values
    for i, pose in enumerate(node.poses_around_ball):
        node.get_logger().info(f"Publishing Pose {i+1}...")
        node.publish_pose(pose)
        while not node.is_pose_close(node.current_arm_pose, pose):
            rclpy.spin_once(node, timeout_sec=1.0)  # <<< process ROS messages
        average_hsv_new = node.visualDetector.average_hsv_masked(node.masked_image_surround)
        node.get_logger().info(f"Average HSV: {average_hsv_new}")
        average_hsv.append(average_hsv_new)
        refl_mean_new, refl_var_new = node.visualDetector.extract_reflectance(node.masked_image_surround, node.mask_surround)
        node.get_logger().info(f"Reflectance Mean: {refl_mean_new}, Variance: {refl_var_new}")
        refl_mean.append(refl_mean_new)
        refl_var.append(refl_var_new)

    # Go above the ball and close the gripper
    pose_above_ball = [world_coords_ball[0, 0], world_coords_ball[1, 0], world_coords_ball[2, 0] + 0.03,
                       1.0, 0.0, 0.0, 0.0]
    pose_ball = [world_coords_ball[0, 0], world_coords_ball[1, 0], world_coords_ball[2, 0] - 0.0127,
                 1.0, 0.0, 0.0, 0.0]
    node.poses_to_ball.append(pose_above_ball)
    # node.poses_to_ball.append(pose_ball)
    # node.get_logger().info(f"Ball pose: {pose_ball}")

    for i, pose in enumerate(node.poses_to_ball):
        node.get_logger().info(f"Publishing Pose {i+1}...")
        node.publish_pose(pose)
        while not node.is_pose_close(node.current_arm_pose, pose, position_threshold=0.005, orientation_threshold=0.005):
            rclpy.spin_once(node, timeout_sec=1.0)  # <<< process ROS messages
    node.get_logger().info("Moving arm to 3cm above ball...")

    # Save the HSV and reflectance values to a csv file
    df = pd.DataFrame({
        'HSV': average_hsv,
        'Reflectance Mean': refl_mean,
        'Reflectance Variance': refl_var
    })
    df.to_csv('hsv_reflectance_data.csv', index=False)
    node.get_logger().info("HSV and reflectance data saved to hsv_reflectance_data.csv")

    node.get_logger().info("Closing gripper...")
    node.publish_gripper_position(1.0)    
    
    # Reset all forces to 0
    node.FT_force_x = 0.0
    node.FT_force_y = 0.0
    node.FT_force_z = 0.0

    # Calculate torque magnitude
    # torque_magnitude = math.sqrt(self.FT_torque_x**2 + self.FT_torque_y**2 + self.FT_torque_z**2)
    
    # Log the force data (components and magnitude)
    # self.get_logger().info(f"Force: [{self.FT_force_x:.2f}, {self.FT_force_y:.2f}, {self.FT_force_z:.2f}] N")
    # self.get_logger().info(f"Force magnitude: {force_magnitude:.2f} N")
    while node.stress_stop_counter < 5:
        force_magnitude = math.sqrt(node.FT_force_x**2 + node.FT_force_y**2 + node.FT_force_z**2)
        node.get_logger().info(f"Force Magnitude: {force_magnitude:.2f} N")

        if node.contact_pose is not None:
            displacement = node.contact_pose.position.z - node.current_arm_pose.position.z
            node.get_logger().info(f"Displacement: {displacement:.2f} m")

            width = 5.1 / 1000
            length = 32.2 / 1000
            area = width * length
            stress = force_magnitude / area
            strain = displacement / (2 * 0.0508)  # original ball length of 0.0508m
            node.strain.append(strain)
            node.stress.append(stress)
        
        # Log the torque data (components and magnitude)
        # node.get_logger().info(f"Torque: [{node.FT_torque_x:.2f}, {node.FT_torque_y:.2f}, {node.FT_torque_z:.2f}] Nm")
        # node.get_logger().info(f"Torque magnitude: {torque_magnitude:.2f} Nm")

        if node.contact_pose is None:
            if force_magnitude > 0.4:
                node.contact_pose = node.current_arm_pose
                node.get_logger().info("Contact detected.")
            else:
                node.get_logger().info("No contact detected.")
                # move the robot down
                node.get_logger().info("Current pose: "
                                        f"[{node.current_arm_pose.position.x:.2f}, "
                                        f"{node.current_arm_pose.position.y:.2f}, "
                                        f"{node.current_arm_pose.position.z:.2f}]")
                pose = [node.current_arm_pose.position.x,
                                    node.current_arm_pose.position.y,
                                    node.current_arm_pose.position.z - 0.007,
                                    node.current_arm_pose.orientation.x,
                                    node.current_arm_pose.orientation.y,
                                    node.current_arm_pose.orientation.z,
                                    node.current_arm_pose.orientation.w]
                node.publish_pose(pose)
                while not node.is_pose_close(node.current_arm_pose, pose, position_threshold=0.001):
                    rclpy.spin_once(node, timeout_sec=1.0)  # <<< process ROS messages
        else:
            if force_magnitude < 0.7:
                node.get_logger().info("Force is below 1.0N, moving robot 0.003 m down.")
                pose = [node.current_arm_pose.position.x,
                                    node.current_arm_pose.position.y,
                                    node.current_arm_pose.position.z - 0.003, 
                                    node.current_arm_pose.orientation.x,
                                    node.current_arm_pose.orientation.y,
                                    node.current_arm_pose.orientation.z,
                                    node.current_arm_pose.orientation.w]
                node.publish_pose(pose)
                while not node.is_pose_close(node.current_arm_pose, pose, position_threshold=0.001):
                    rclpy.spin_once(node, timeout_sec=1.0)  # <<< process ROS messages
            elif force_magnitude < 1.5:
                node.get_logger().info("Force is between 1N and 1.75N, moving robot 0.001m down.")
                pose = [node.current_arm_pose.position.x,
                                    node.current_arm_pose.position.y,
                                    node.current_arm_pose.position.z - 0.001,
                                    node.current_arm_pose.orientation.x,
                                    node.current_arm_pose.orientation.y,
                                    node.current_arm_pose.orientation.z,
                                    node.current_arm_pose.orientation.w]
                node.publish_pose(pose)
                start_time = node.get_clock().now()  # <<< Get ROS2 time now
                while not node.is_pose_close(node.current_arm_pose, pose, position_threshold=0.001):
                    rclpy.spin_once(node, timeout_sec=1.0)

                    # Check if more than 5 seconds have passed
                    elapsed_time = node.get_clock().now() - start_time
                    if elapsed_time.nanoseconds * 1e-9 > 5.0:
                        node.get_logger().warn("Timeout: Robot did not reach pose in 5 seconds. Setting final pose.")
                        if node.final_pose is None:
                            node.final_pose = node.current_arm_pose
                            node.get_logger().info("Final pose set after timeout.")
                            node.stress_stop_counter = 5
                        break
            else:
                node.get_logger().info("Force is above 1.75N, stopped the robot.")
                node.stress_stop_counter += 1
                if node.final_pose is None:
                    node.final_pose = node.current_arm_pose
                    node.get_logger().info("Final pose set.")
    
    node.get_logger().info("Moving back to initial position...")
    init_pose = [
        node.init_arm_pose.position.x,
        node.init_arm_pose.position.y,
        node.init_arm_pose.position.z,
        node.init_arm_pose.orientation.x,
        node.init_arm_pose.orientation.y,
        node.init_arm_pose.orientation.z,
        node.init_arm_pose.orientation.w
    ]
    node.publish_pose(pose_above_ball)
    node.publish_pose(init_pose)

    node.publish_gripper_position(0.0)  # Open the gripper

    # Save the strain and stress data to a csv file
    data = {
        'strain': node.strain,
        'stress': node.stress
    }
    df = pd.DataFrame(data)
    df.to_csv('strain_stress_data.csv', index=False)
    node.get_logger().info("Strain and stress data saved to strain_stress_data.csv.")

    # node.get_logger().info("Opening gripper...")
    # node.publish_gripper_position(0.0)

    # # Publish the ball world coordinates
    # ball_pose = Pose()
    # ball_pose.position.x = world_coords_ball[0, 0]
    # ball_pose.position.y = world_coords_ball[1, 0]
    # ball_pose.position.z = world_coords_ball[2, 0]
    # ball_pose.orientation.x = 0.0
    # ball_pose.orientation.y = 0.0
    # ball_pose.orientation.z = 0.0
    # ball_pose.orientation.w = 1.0
    # node.ball_world_coords_pub.publish(ball_pose)
    # node.get_logger().info(f"Published ball world coordinates: {ball_pose}")

    node.get_logger().info("All actions done. Shutting down.")

    node.destroy_node()
    rclpy.shutdown()

    matplotlib.use('Agg')          # guarantees no GUI errors

    df = pd.read_csv('strain_stress_data.csv')

    # Scatter plot
    plt.scatter(df['strain'], df['stress'], color='blue', label='Data')

    # # Line of best fit
    # X = df['strain'].values.reshape(-1, 1)
    # y = df['stress'].values
    # model = LinearRegression()
    # model.fit(X, y)
    # y_pred = model.predict(X)

    # # Plot the line
    # plt.plot(df['strain'], y_pred, color='red', label='Best Fit Line')

    # # R^2 score
    # r2 = r2_score(y, y_pred)
    # plt.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=plt.gca().transAxes,
    #         fontsize=10, verticalalignment='top')

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