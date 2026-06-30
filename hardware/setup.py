from glob import glob
from setuptools import find_packages, setup

package_name = 'hardware'

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
    description='Hardware interface nodes for the autonomous driving competition vehicle.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'camera_node = hardware.camera_node:main',
            'lidar_node = hardware.lidar_node:main',
            'ultrasonic_node = hardware.ultrasonic_node:main',
            'manual_controller_node = hardware.manual_controller_node:main',
            'camera_viewer_node = hardware.camera_viewer_node:main',
            'lidar_viewer_node = hardware.lidar_viewer_node:main',
            'ultrasonic_viewer_node = hardware.ultrasonic_viewer_node:main',
            'camera_calibration_node = hardware.camera_calibration_node:main',
            'camera_pose_check_node = hardware.camera_pose_check_node:main',
        ],
    },
)
