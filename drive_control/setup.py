from glob import glob
from setuptools import find_packages, setup

package_name = 'drive_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hailab',
    maintainer_email='hailab@example.com',
    description='Drive control placeholder.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'drive_control_node = drive_control.drive_control_node:main',
        ],
    },
)
