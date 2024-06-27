from setuptools import setup, find_packages

setup(
    name='generate_captions',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'Pillow',
        'salesforce-lavis',
    ],
    entry_points={
        'console_scripts': [
            'generate-captions=generate_captions.generate_captions:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A script to generate captions for images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-repository',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
