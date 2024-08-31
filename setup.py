import os.path
from setuptools import setup, find_packages


home = os.path.dirname(os.path.abspath(__file__))

__version__ = '0.0.0'
exec(open(os.path.join(home, 'vklearn/version.py')).read())

with open(os.path.join(home, 'README.md')) as f:
    readme = f.read()

with open(os.path.join(home, 'requirements.txt')) as f:
    requirements = [item.strip() for item in f]


setup(
    name='vikit-learn',
    version=__version__,
    author='jojowee',
    description='A computer vision toolkit that is easy-to-use and based on deep learning.',
    long_description=readme,
    license='Apache-2.0 License',
    url='https://github.com/bxt-kk/vikit-learn',
    project_urls={
       'Source Code': 'https://github.com/bxt-kk/vikit-learn', 
    },
    packages=find_packages(),
    package_data={'vklearn':[
        'datasets/ms_coco_classnames.json',
    ]},
    install_requires=requirements,
    entry_points = {
        'console_scripts': [
            'vkl-clf-cfg=toolkit.clf_cfg:entry',
            'vkl-clf-cli=toolkit.clf_cli:entry',
        ],
    }
)
