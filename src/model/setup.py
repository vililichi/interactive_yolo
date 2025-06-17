from setuptools import find_packages, setup

package_name = 'interactive_yolo_model'

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
            'test_node = interactive_yolo_model.test_node:main',
            'ponctual_inference_node = interactive_yolo_model.ponctual_inference_node:main',
            'model_node = interactive_yolo_model.model_node:main',
            'tool_classes_distances_histogram = interactive_yolo_model.tools.classes_distances_histogram.classes_distances_histogram:main',
            'tool_evaluate_model = interactive_yolo_model.tools.evaluate_model.evaluate_model:main',
            'tool_generate_engine_models = interactive_yolo_model.tools.generate_engine_models.generate_engine_models:main'
        ],
    },
)
