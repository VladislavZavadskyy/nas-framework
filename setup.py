from setuptools import setup, find_packages
import nasframe

setup(
    name='nasframe',
    description='Neural Architecture Search Framework',
    version=nasframe.__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'torch',
        'flatten_dict',
        'numpy',
        'scikit-learn',
        'scipy',
        'pyyaml',
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
        nas=nasframe.scripts.cli:nas
    ''',
)