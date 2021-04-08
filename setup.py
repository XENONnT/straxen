import setuptools

# Get requirements from requirements.txt, stripping the version tags
with open('requirements.txt') as f:
    requires = [
        r.split('/')[-1] if r.startswith('git+') else r
        for r in f.read().splitlines()]

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='straxen',
                 version='0.16.0',
                 description='Streaming analysis for XENON',
                 author='Straxen contributors, the XENON collaboration',
                 url='https://github.com/XENONnT/straxen',
                 long_description=readme + '\n\n' + history,
                 long_description_content_type="text/markdown",
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + [
                     'tensorflow',
                     'pytest',
                     'hypothesis',
                     'holoviews',
                     'flake8',
                     'pytest-cov',
                     'coveralls',
                     'xarray',
                     'datashader',
                     'boltons',
                     'ipython',
                 ],
                 python_requires=">=3.6",
                 extras_require={
                     'docs': ['sphinx',
                              'sphinx_rtd_theme',
                              'nbsphinx',
                              'recommonmark',
                              'graphviz'],
                     'microstrax': ['hug'],
                 },
                 scripts=['bin/bootstrax', 'bin/straxer', 'bin/fake_daq',
                          'bin/microstrax', 'bin/ajax',
                          'bin/refresh_raw_records'],
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
