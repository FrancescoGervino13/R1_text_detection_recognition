from setuptools import find_packages, setup

package_name = 'mmocr_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','mmocr'],
    zip_safe=True,
    maintainer='fgervino',
    maintainer_email='fgervino@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'mmocr_node = mmocr_ros.mmocr_node:main',
        'image_publisher_node = mmocr_ros.image_publisher_node:main',
        'mmocr_node_try = mmocr_ros.mmocr_node_try:main',
        'mmocr_node_process = mmocr_ros.mmocr_node_process:main',
        'image_subscriber = image_subscriber:main',
        'detection_node = detection_node:main',
        'recognition_node = recognition_node:main',
        ],
    },
)
