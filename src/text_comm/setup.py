from setuptools import setup

package_name = 'text_comm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='frenci',
    maintainer_email='francesco.gervino@iit.it',
    description='A simple text communication package using ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'text_publisher = text_comm.text_publisher:main',
            'text_subscriber = text_comm.text_subscriber:main',
            'publish_gpt4 = text_comm.publish_gpt4:main',
            'subscribe_gpt4 = text_comm.subscribe_gpt4:main',
        ],
    },
)