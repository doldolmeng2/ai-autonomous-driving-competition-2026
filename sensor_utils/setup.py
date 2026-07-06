from glob import glob
from setuptools import find_packages, setup

package_name = 'sensor_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hailab',
    maintainer_email='hailab@example.com',
    description='Sensor visualization, calibration, bag, and controller utility nodes.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'joy_to_motor_node = sensor_utils.joy_to_motor_node:main',
            'controller_viewer_node = sensor_utils.controller_viewer_node:main',
            'camera_viewer_node = sensor_utils.camera_viewer_node:main',
            'lidar_viewer_node = sensor_utils.lidar_viewer_node:main',
            'ultrasonic_viewer_node = sensor_utils.ultrasonic_viewer_node:main',
            'camera_calibration_node = sensor_utils.camera_calibration_node:main',
            'camera_pose_check_node = sensor_utils.camera_pose_check_node:main',
            'hsv_tuner_node = sensor_utils.hsv_tuner_node:main',
        ],
    },
)
