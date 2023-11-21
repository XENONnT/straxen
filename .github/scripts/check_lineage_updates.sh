#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <first_branch_name> <second_branch_name> <output folder path> <run_id>"
    exit 1
fi

old_branch="$1"
new_branch="$2"
output_path="$3"
run_id="$4"

cd straxen
git checkout "$old_branch"

get_old_plugins=$(cat<<'EOF'
import strax, straxen
import numpy as np
import sys
import json

output_path = sys.argv[3]

st = straxen.contexts.xenonnt_online(output_folder=output_path)
st.storage.append(strax.DataDirectory(output_path, readonly = True))

version_hash_dict_old = strax.utils.version_hash_dict(st)
with open("version_hash_dict_old.json", 'w') as jsonfile:
    json.dump(version_hash_dict_old, jsonfile)
EOF
)

python3 -c "$get_old_plugins" "$@"


git checkout "$new_branch"

get_new_plugins=$(cat<<'EOF'
import strax, straxen
import numpy as np
import sys
import json

output_path = sys.argv[3]

run_id = sys.argv[4]

st = straxen.contexts.xenonnt_online(output_folder=output_path)
st.storage.append(strax.DataDirectory(output_path, readonly = True))

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
print("\nDeleted plugins:")
for p in updated_plugins_dict['deleted']:
    print(f"    - {p}")
print('\n')

#Now print the info for the added plugins
bad_field_info_added = strax.utils.bad_field_info(st, run_id, updated_plugins_dict['added'])

for p in bad_field_info_added:
    print(f"\nNew plugin '{p}' has the following bad field fractions:")
    for c in bad_field_info_added[p]:
        if bad_field_info_added[p][c]>0:
            #Don't print the mean values (unless the column name literally starts with mean)
            if (not c.startswith('mean')) or (c.startswith('mean_mean')):
                print(f"    - {c}: {bad_field_info_added[p][c]}")
print('\n')

##################################### Comparing differences to changed plugins #####################################

lowest_level_changed_plugins = strax.utils.lowest_level_plugins(st, updated_plugins_dict['changed'])

#See the nan field fractions + mean of each field
new_changed_plugin_bad_info = strax.utils.bad_field_info(st, run_id, lowest_level_changed_plugins)

with open("new_changed_plugin_bad_info.json", 'w') as jsonfile:
    json.dump(new_changed_plugin_bad_info, jsonfile)
print("Finish writing to file")
#affected means the plugins which directly depend on the lowest level changed plugins
affected_changed_plugins = strax.utils.directly_depends_on(st,
    lowest_level_changed_plugins,
    updated_plugins_dict['changed'])

new_affected_plugin_bad_info = strax.utils.bad_field_info(st, run_id, affected_changed_plugins)
with open("new_affected_plugin_bad_info.json", 'w') as jsonfile:
    json.dump(new_affected_plugin_bad_info, jsonfile)
EOF
)

python3 -c "$get_new_plugins" "$@"

git checkout "$old_branch"

compare_plugins=$(cat<<'EOF'
import strax, straxen
import numpy as np
import sys
import json

output_path = sys.argv[3]

run_id = sys.argv[4]
#'025423'

st = straxen.contexts.xenonnt_online(output_folder=output_path)
st.storage.append(strax.DataDirectory(output_path, readonly = True))

#Load in the updated plugins dict
with open("plugin_update_comparison.json", 'r') as jsonfile:
    updated_plugins_dict = json.load(jsonfile)

#Load the bad fields info of the newly changed plugins (remember, we're back to the old branch)
with open("new_changed_plugin_bad_info.json", 'r') as jsonfile:
    new_changed_plugin_bad_info = json.load(jsonfile)

#Load the affected plugins bad info
with open("new_affected_plugin_bad_info.json", 'r') as jsonfile:
    new_affected_plugin_bad_info = json.load(jsonfile)

##################### Now compute the same for the old version of the plugins #####################
old_changed_plugin_bad_info = strax.utils.bad_field_info(st, run_id, list(new_changed_plugin_bad_info.keys()))
old_affected_plugin_bad_info = strax.utils.bad_field_info(st, run_id, list(new_affected_plugin_bad_info.keys()))

###Now report the differences
#Lowest level plugins
all_plugin_change_info = {"Lowest Levels":{'old':old_changed_plugin_bad_info,
    'new':new_changed_plugin_bad_info},
    "Affected":{'old':old_affected_plugin_bad_info,
    'new':new_affected_plugin_bad_info}}

for level in ['Lowest Levels', 'Affected']:
    print(f"#################### {level} Plugins ####################")

    for p in all_plugin_change_info[level]['old']:
        print(f"Change report for '{p}':")
        data_types_old = np.array(list(all_plugin_change_info[level]['old'][p].keys()))[::2]
        data_types_new = np.array(list(all_plugin_change_info[level]['new'][p].keys()))[::2]

        all_data_types = np.unique(np.concatenate([data_types_old, data_types_new]))
        for d in all_data_types:
            if d not in data_types_old:
                print(f"    - New column {d} added")
            elif d not in data_types_new:
                print(f"    - Column {d} deleted")
            else:
                if (all_plugin_change_info[level]['old'][p][d] != all_plugin_change_info[level]['new'][p][d]):
                    print(f"    - {d} bad fraction changed from: {all_plugin_change_info[level]['old'][p][d]} -> {all_plugin_change_info[level]['new'][p][d]}")
                if (all_plugin_change_info[level]['old'][p][f'mean_{d}'] != all_plugin_change_info[level]['new'][p][f'mean_{d}']):
                    print(f"    - {d} mean value changed from: {all_plugin_change_info[level]['old'][p][f'mean_{d}']} -> {all_plugin_change_info[level]['new'][p][f'mean_{d}']}")
        print("All other columns remained the same\n")
    print('\n')
EOF
)

python3 -c "$compare_plugins" "$@"

#Remove the temporary dictionaries that were created to deal with switching branches
rm new_affected_plugin_bad_info.json
rm new_changed_plugin_bad_info.json
rm plugin_update_comparison.json
rm version_hash_dict_new.json
rm version_hash_dict_old.json
