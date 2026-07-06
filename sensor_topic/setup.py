from glob import glob
from setuptools import find_packages, setup

package_name = 'sensor_topic'

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
    description='Sensor and controller topic publisher nodes.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'camera_node = sensor_topic.camera_node:main',
            'lidar_node = sensor_topic.lidar_node:main',
            'ultrasonic_node = sensor_topic.ultrasonic_node:main',
            'controller_node = sensor_topic.controller_node:main',
        ],
    },
)
