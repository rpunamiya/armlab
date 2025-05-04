#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # Add a delay before starting Gazebo to let ROS initialize
    initial_delay = ExecuteProcess(
        cmd=["sleep", "5"],
        name="initial_delay"
    )

    # Include the xarm7 MoveIt+Gazebo launch
    xarm_moveit_gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xarm_moveit_config'),
                'launch',
                'xarm7_moveit_gazebo.launch.py'
            )
        ),
        launch_arguments={
            'add_gripper': 'true',
            'add_realsense_d435i': 'true',
            'add_ft_sensor': 'true',
        }.items()
    )

    # Launch xarm_pose_commander with a delay
    xarm_pose_commander_node = Node(
        package='me314_pkg',  
        executable='xarm_commander.py',    
        output='screen',
        parameters=[{'use_sim': True}] 
    )

    # Add a delay before starting the commander
    delayed_commander = TimerAction(period=10.0, actions=[xarm_pose_commander_node])

    block_spawn = TimerAction(
        period=15.0,  # give Gazebo time to start
        actions=[Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            output='screen',
            arguments=[
                '-file', os.path.join(
                    get_package_share_directory('me314_pkg'),
                    'gazebo_models', 'red_block.sdf'
                ),
                '-entity', 'red_block',
                '-x', '-0.35', '-y', '-0.85', '-z', '1.021'
            ],
            parameters=[{'use_sim_time': True}],
        )]
    )

    green_square_spawn = TimerAction(
        period=20.0,  # give Gazebo time to start
        actions=[Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            output='screen',
            arguments=[
                '-file', os.path.join(
                    get_package_share_directory('me314_pkg'),
                    'gazebo_models', 'green_square.sdf'
                ),
                '-entity', 'green_square',
                '-x', '0.03', '-y', '-0.9', '-z', '1.021'
            ],
            parameters=[{'use_sim_time': True}],
        )]
    )

    return LaunchDescription([
        initial_delay,
        xarm_moveit_gazebo_launch,
        delayed_commander,
        block_spawn,
        green_square_spawn
    ])