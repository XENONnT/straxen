import setuptools


def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires


# Get requirements from requirements.txt, stripping the version tags
requires = open_requirements('requirements.txt')
tests_requires = open_requirements('extra_requirements/requirements-tests.txt')
doc_requirements = open_requirements('extra_requirements/requirements-docs.txt')

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(name='straxen',
                 version='0.18.8',
                 description='Streaming analysis for XENON',
                 author='Straxen contributors, the XENON collaboration',
                 url='https://github.com/XENONnT/straxen',
                 long_description=readme + '\n\n' + history,
                 long_description_content_type="text/markdown",
                 setup_requires=['pytest-runner'],
                 install_requires=requires,
                 tests_require=requires + tests_requires,
                 python_requires=">=3.6",
                 extras_require={
                     'docs': doc_requirements,
                     'microstrax': ['hug'],
                 },
                 scripts=[
                     'bin/bootstrax',
                     'bin/straxer',
                     'bin/fake_daq',
                     'bin/microstrax',
                     'bin/ajax',
                     'bin/refresh_raw_records',
                 ],
                 packages=setuptools.find_packages() + ['extra_requirements'],
                 package_dir={'extra_requirements': 'extra_requirements'},
                 package_data={'extra_requirements': ['requirements-docs.txt',
                                                      'requirements-tests.txt']},
                 classifiers=[
                     'Development Status :: 5 - Production/Stable',
                     'License :: OSI Approved :: BSD License',
                     'Natural Language :: English',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Topic :: Scientific/Engineering :: Physics',
                 ],
                 zip_safe=False)
