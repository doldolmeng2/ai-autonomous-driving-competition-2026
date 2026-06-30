from setuptools import find_packages, setup

package_name = 'lane_offset'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hailab',
    maintainer_email='hailab@example.com',
    description='Lane offset calculation placeholders.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'timed_lane_offset_node = lane_offset.timed_lane_offset_node:main',
            'mission_lane_offset_node = lane_offset.mission_lane_offset_node:main',
        ],
    },
)
