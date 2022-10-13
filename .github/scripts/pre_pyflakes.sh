#!/bin/bash
# Pyflakes does not like the way we do __all__ += []. This simple script
# Changes all the files in straxen to abide by this convention and
# removes the lines that have such a signature.

# Fixes 
# AttributeError: 'Binding' object has no attribute 'names'
# "pyflakes" failed during execution due to "'Binding' object has no attribute 'names'"
# Run flake8 with greater verbosity to see more details
start="$(pwd)"
echo $start

cd straxen
sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py

cd plugins
sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py

cd ../analyses
sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py

cd ../storage
sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py

cd $start
echo "done"
