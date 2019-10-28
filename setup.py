import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [x.strip().split('=')[0]
                for x in f.readlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='nEXO_strax',
                 version='0.1.0',
                 description='Streaming analysis for nEXO',
                 author='The nEXO collaboration',
                 url='https://github.com/nEXO-collaboration/nEXO_strax',
                 long_description=readme + '\n\n' + history,
                 long_description_content_type="text/markdown",
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + ['pytest'],
                 python_requires=">=3.6",
                 extras_require={
                     'docs': ['sphinx',
                              'sphinx_rtd_theme',
                              'nbsphinx',
                              'recommonmark',
                              'graphviz'],
                 },
                 scripts=['bin/bootstrax', 'bin/straxer'],
                 packages=setuptools.find_packages(),
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'License :: OSI Approved :: BSD License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.6',
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Topic :: Scientific/Engineering :: Physics',
                 ],
                 zip_safe=False)
