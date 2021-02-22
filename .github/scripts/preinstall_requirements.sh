#!/bin/bash

echo "download requirements from base_environment"
wget -O pre_requirements.txt https://raw.githubusercontent.com/XENONnT/base_environment/master/requirements.txt &> /dev/null
echo "select important dependencies for strax(en)"
grep 'numpy\|numba\|tensorflow\|blosc\|scikit-learn' pre_requirements.txt &> sel_pre_requirements.txt
echo "Will pre-install:"
cat sel_pre_requirements.txt
echo "Start preinstall and rm pre-requirements:"
pip install -r sel_pre_requirements.txt
rm pre_requirements.txt sel_pre_requirements.txt
