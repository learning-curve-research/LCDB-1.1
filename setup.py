from setuptools import setup, find_packages

setup(
    name='lcdb1.1',
    version='0.1.0',
    description='A description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/lcdb1.1',
    packages=find_packages(include=['lcdb_function', 'lcdb_function.*']),
    install_requires=[
        # List your package dependencies here
        # 'numpy>=1.19.2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
