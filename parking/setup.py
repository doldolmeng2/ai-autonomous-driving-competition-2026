from glob import glob
from setuptools import find_packages, setup

package_name = 'parking'

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
    description='Parking mission placeholder.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'parking_node = parking.parking_node:main',
        ],
    },
)
