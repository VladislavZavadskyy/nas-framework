from setuptools import setup, find_packages

setup(
    name='nasframe',
    description='Neural Architecture Search Framework',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'torch',
        'flatten_dict',
        'numpy',
        'scikit-learn',
        'scipy',
        'pillow',
        'tensorboardX',
        'pygraphviz',
    ],
    extras_require={
        'text': [
            'stop-words',
            'pyenchant',
            'pandas',
            'jellyfish',
        ],
        'docs': [
            'sphinx-rtd-theme'
        ]
    },
    entry_points='''
        [console_scripts]
        toxic_nas=nasframe.scripts.train_toxic:cli
    ''',
)