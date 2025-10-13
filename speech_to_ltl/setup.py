from setuptools import setup, find_packages

package_name = 'speech_to_ltl'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/ltl_translate.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yiyi',
    maintainer_email='yiyi@ucsd.edu',
    description='LTL Translate Node (ROS 2)',
    license='TODO',
    entry_points={
        'console_scripts': [
            'ltl_translate_node = speech_to_ltl.ltl_translate_node:main',
        ],
    },
)
