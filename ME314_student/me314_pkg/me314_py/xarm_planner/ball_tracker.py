#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import csv
import os
import pandas as pd
import ast

class BallTrackerOffline(Node):
    def __init__(self):
        super().__init__('ball_tracker_offline')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/standalone/image_raw',
            self.image_callback,
            qos_profile_sensor_data)

        # Configuration
        self.capture_duration = 5.0  # seconds
        self.output_video_path = 'bouncing_ball.avi'
        self.output_csv_path   = 'ball_height.csv'
        self.fps = 30
        self.frame_size = (640, 480)  # match your camera resolution

        # Ball color HSV range (example: yellow)
        self.hsv_mean, self.hsv_tol = self.load_hsv_from_csv('hsv_reflectance_data.csv')

        # State
        self.frames = []
        self.start_time = None
        self.recording = True

        self.get_logger().info('Recording for 5 seconds...')
    
    def load_hsv_from_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)

            # Parse HSV strings like "(118.6, 238.0, 90.9)" into float tuples
            hsv_values = df.iloc[:, 0].apply(lambda x: ast.literal_eval(x)).tolist()

            # Convert to NumPy for easy averaging
            hsv_array = np.array(hsv_values)  # shape: (N, 3)
            avg_hsv = np.mean(hsv_array, axis=0)
            std_hsv = np.std(hsv_array, axis=0)

            avg_hsv = [10, 200, 100]
            self.get_logger().info(f"Average HSV from CSV: {avg_hsv}")
            std_hsv = (10, 50, 50)
            self.get_logger().info(f"Standard Deviation HSV from CSV: {std_hsv}")
            return tuple(int(round(x)) for x in avg_hsv), tuple(int(round(x)) for x in std_hsv)
        except Exception as e:
            self.get_logger().error(f"Error loading HSV values from CSV: {e}")
            return (25, 180, 180), (15, 70, 70)  # Default values
    
    def image_callback(self, msg):
        if not self.recording:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if self.start_time is None:
            self.start_time = now

        # Convert ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frames.append(frame)

        if now - self.start_time >= self.capture_duration:
            self.recording = False
            self.get_logger().info(f"Captured {len(self.frames)} frames. Saving...")
            self.save_video()
            self.process_video()
            self.get_logger().info("Done. Shutting down.")
            rclpy.shutdown()

    def save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, self.frame_size)
        for frame in self.frames:
            resized = cv2.resize(frame, self.frame_size)
            writer.write(resized)
        writer.release()
        self.get_logger().info(f"Video saved to {self.output_video_path}")

    def process_video(self):
        cap = cv2.VideoCapture(self.output_video_path)
        heights = []
        times = []

        # depth_frames = np.load('depth_video.npy')  # shape: (N, H, W)
        intrinsics = {'fx': 609.4088745117188, 'fy': 609.080078125, 'cx': 335.96954345703125, 'cy': 247.07424926757812}

        for i in range(len(self.frames)):
            ret, frame = cap.read()
            if not ret:
                break
            t = i / self.fps
            h = self.extract_ball_height(frame, intrinsics)
            if h is not None:
                heights.append(h)
                times.append(t)
        
        cap.release()
        with open(self.output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_sec', 'height_m'])
            for t, h in zip(times, heights):
                writer.writerow([f"{t:.3f}", f"{h:.4f}"])
        self.get_logger().info(f"Height CSV saved to {self.output_csv_path}")

    def extract_ball_height(self, frame, intrinsics):
        cv2.imwrite("test_frame.png", frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.subtract(self.hsv_mean, self.hsv_tol)
        upper = np.add(self.hsv_mean, self.hsv_tol)
        mask  = cv2.inRange(hsv, lower, upper)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        overlay = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite("test_overlay.png", overlay)


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().info("No contours found.")
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 200:
            return None
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
        u = int(M['m10'] / M['m00'])  # pixel column
        v = int(M['m01'] / M['m00'])  # pixel row

        # Get depth at (u, v)
        # if depth_frame.shape != frame.shape[:2]:
        #     depth_frame = cv2.resize(depth_frame, (frame.shape[1], frame.shape[0]))
        # Z = depth_frame[v, u] / 1000.0  # mm to meters
        # if Z == 0:
        #     return None
        Z = 0.52

        fy = intrinsics['fy']
        cy = intrinsics['cy']

        Y = (v - cy) * Z / fy  # vertical height from pinhole model

        return Y

def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerOffline()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()