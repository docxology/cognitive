from setuptools import setup, find_packages

setup(
    name="ant_colony",
    version="1.0.0",
    description="Multi-agent ant colony simulation",
    author="Your Name",
    packages=find_packages(),
    package_data={
        'ant_colony': ['config/*.yaml']
    },
    install_requires=[
        'numpy>=1.26.0',
        'matplotlib>=3.8.0',
        'networkx>=3.0',
        'pyyaml>=6.0',
        'noise>=1.2.2'  # For terrain generation
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'ant-colony=ant_colony.main:main',
        ],
    },
) 