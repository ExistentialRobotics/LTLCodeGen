from setuptools import setup, find_packages

package_name = 'label_map'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ros_label_map.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yiyi',
    maintainer_email='yiyi@ucsd.edu',
    description='Label Map Node (ROS 2)',
    license='TODO',
    entry_points={
        'console_scripts': [
            'label_map_ros = label_map.label_map_ros:main',
        ],
    },
)
