
#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <first_branch_name> <second_branch_name>"
    exit 1
fi

old_branch="$1"
new_branch="$2"

cd straxen
git checkout "$old_branch"

python3 - <<END
import strax, straxen, cutax
import numpy as np

st = straxen.contexts.xenonnt_online(output_folder='/scratch/midway2/jyangqi/strax_data')
st.storage.append(strax.DataDirectory(scratch_folder, readonly = True))
st.storage = [st.storage[-1], st.storage[-3]]

version_hash_dict_old = strax.utils.version_hash_dict(st)
with open("version_hash_dict_old.json", 'w') as jsonfile:
    json.dump(version_hash_dict_old, jsonfile)
END

git checkout "$new_branch"

python3 - <<END
import strax, straxen, cutax
import numpy as np

run_id = '025423'

st = straxen.contexts.xenonnt_online(output_folder='/scratch/midway2/jyangqi/strax_data')
st.storage.append(strax.DataDirectory(scratch_folder, readonly = True))
st.storage = [st.storage[-1], st.storage[-3]]

#Make the new version-hash dictionary
version_hash_dict_new = strax.utils.version_hash_dict(st)
with open("version_hash_dict_new.json", 'w') as jsonfile:
    json.dump(version_hash_dict_new, jsonfile)

#... and open the old one
with open("version_hash_dict_old.json", 'r') as jsonfile:
    version_hash_dict_old = json.load(jsonfile)

#... to compare the changes
updated_plugins_dict = strax.utils.updated_plugins(version_hash_dict_old, version_hash_dict_new)
with open("plugin_update_comparison.json", 'w') as jsonfile:
    json.dump(updated_plugins_dict, jsonfile)

#Print the deleted plugins:
print("Deleted plugins:")
for p in updated_plugins_dict['deleted']:
    print(f"    - {p}")

print('\n')

#Now print the info for the added plugins
bad_field_info_added = strax.utils.bad_field_info(st, run_id, updated_plugins_dict['added'])

for p in bad_field_info_added:
    print(f"New plugin '{p}' has the following bad field fractions:")
    for c in bad_field_info_added[p]:
        if bad_field_info_added[p][c]>0:
            print(f"    - {c}: {bad_field_info_added[p][c]}")

END


