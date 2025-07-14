from setuptools import find_packages, setup

package_name = 'paddleocr_ros'

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
        'det_rec = det_rec:main',
        ],
    },
)
