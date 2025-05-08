#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('standalone_realsense_publisher')
        self.declare_parameter('serial_no', '317222074068')
        serial = self.get_parameter('serial_no').get_parameter_value().string_value

        if not serial:
            self.get_logger().error("serial_no parameter is required.")
            rclpy.shutdown()
            return

        # RealSense config
        ctx = rs.context()
        devices = ctx.query_devices()
        if not any(d.get_info(rs.camera_info.serial_number) == serial for d in devices):
            self.get_logger().error(f"Camera {serial} not found.")
            rclpy.shutdown()
            return

        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(cfg)

        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/standalone/image_raw', 10)
        self.info_pub  = self.create_publisher(CameraInfo, '/standalone/camera_info', 10)
        self.depth_pub = self.create_publisher(Image, '/standalone/depth/image_raw', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/standalone/depth/camera_info', 10)

        self.frame_id = 'standalone_camera_color_optical_frame'
        self.timer = self.create_timer(1.0 / 30.0, self.publish_frame)
        self.get_logger().info(f"Publishing image + intrinsics from camera {serial}...")

    def publish_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return

        # Convert image
        frame = np.asanyarray(color_frame.get_data())
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = self.frame_id

        # CameraInfo message
        cam_info = CameraInfo()
        cam_info.header = img_msg.header
        cam_info.height = self.intrinsics.height
        cam_info.width  = self.intrinsics.width
        cam_info.distortion_model = 'plumb_bob'  # or self.intrinsics.model.name.lower()
        cam_info.d = list(self.intrinsics.coeffs)
        cam_info.k = [
            self.intrinsics.fx, 0.0, self.intrinsics.ppx,
            0.0, self.intrinsics.fy, self.intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        cam_info.r = [1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0]
        cam_info.p = [
            self.intrinsics.fx, 0.0, self.intrinsics.ppx, 0.0,
            0.0, self.intrinsics.fy, self.intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        self.image_pub.publish(img_msg)
        self.info_pub.publish(cam_info)

        # Depth image
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
        depth_msg.header = img_msg.header  # same timestamp & frame
        self.depth_pub.publish(depth_msg)

        # Reuse intrinsics for depth (or re-query if different stream profile is needed)
        depth_info = CameraInfo()
        depth_info.header = img_msg.header
        depth_info.height = self.intrinsics.height
        depth_info.width  = self.intrinsics.width
        depth_info.distortion_model = 'plumb_bob'
        depth_info.d = list(self.intrinsics.coeffs)
        depth_info.k = [
            self.intrinsics.fx, 0.0, self.intrinsics.ppx,
            0.0, self.intrinsics.fy, self.intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        depth_info.r = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
        depth_info.p = [
            self.intrinsics.fx, 0.0, self.intrinsics.ppx, 0.0,
            0.0, self.intrinsics.fy, self.intrinsics.ppy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        self.depth_info_pub.publish(depth_info)


    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RealSensePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
