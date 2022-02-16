#!/bin/bash
# Argument 1 tells where to put the file (should be in $HOME or the folder
# where one is testing)

echo "write to $1/pre_apply_function.py"
mkdir "$1"

# Create dummy file for testing with pre_apply_function
cat > "$1/pre_apply_function.py" <<EOF
def pre_apply_function(data, run_id, target):
    print(f'Pre-applying function to {run_id}-{target}')
    return data
EOF
