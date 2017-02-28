from setuptools import setup


def readme():
      with open('README.rst') as f:
            return f.read()

setup(name='robust_tail',
      version='0.1',
      description='Companion package to the paper by Lam and Mottet.',
      long_description=readme(),
      url='http://github.com/cmottet/robust_tail',
      author='Clementine Mottet',
      author_email='cmottet@bu.edu',
      license='MIT',
      packages=['robust_tail'],
      install_requires=['pandas', 'numpy'],
      zip_safe=False)

