from setuptools import find_packages, setup

package_name = 'interactive_yolo_audio'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'torchvision',
        'ultralytics'
    ],
    zip_safe=True,
    maintainer='vililichi',
    maintainer_email='keven0230@live.ca',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stt_node = interactive_yolo_audio.stt_node:main',
            'tts_node = interactive_yolo_audio.tts_node:main'
        ],
    },
)
