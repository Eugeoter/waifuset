from setuptools import setup, find_packages
with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

packages = find_packages()

setup(
    name='waifuset',
    version='0.1',
    packages=packages,
    description='Image caption tools',
    long_description='',
    author='euge',
    author_email='1507064225@qq.com',
    url='https://github.com/Eugeoter/stable-dataset-manager',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)
