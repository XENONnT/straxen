#!/bin/bash

if [ ! -z "$RUNDB_API_URL" ]
then
cat > $HOME/.xenon_config <<EOF
[basic]
logging_level=DEBUG

[RunDB]
rundb_api_url = $RUNDB_API_URL
rundb_api_user = $RUNDB_API_USER_READONLY
rundb_api_password = $RUNDB_API_PASSWORD_READONLY
xent_url = $PYMONGO_URL
xent_user = $PYMONGO_USER
xent_password = $PYMONGO_PASSWORD
xent_database = $PYMONGO_DATABASE
pymongo_url = $PYMONGO_URL
pymongo_user = $PYMONGO_USER
pymongo_password = $PYMONGO_PASSWORD
pymongo_database = $PYMONGO_DATABASE

[scada]
scdata_url = $SCADA_URL
sclastvalue_url = $SCADA_VALUE_URL

sclogin_url = $SCADA_LOGIN_URL
straxen_username = $SCADA_USER
straxen_password = $SCADA_PWD
pmt_parameter_names = no_file_found
EOF
echo "YEAH boy, complete github actions voodoo now made you have access to our database!"
else
 echo "You have no power here! Environment variables are not set, therefore no utilix file will be created"
fi
