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

sub_dirs="$(ls -d */ | grep -v '.py\|plugins')"
for folder in $sub_dirs;
 do cd $folder;
 sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py;
 cd ..;
done

cd plugins
data_kinds="$(ls -d */)"
for folder in $data_kinds;
 do cd $folder;
 sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py;
 cd ..;
done
cd ..

cd legacy/plugins_1t
sed -e '/__all__ +=/ s/^#*/#/' -i ./*.py
cd ../..

cd $start
echo "done"
