#!/bin/bash

cat > $HOME/.xenon_config <<EOF
[RunDB]
rundb_api_url = $RUNDB_API_URL
rundb_api_user = $RUNDB_API_USER_READONLY
rundb_api_password = $RUNDB_API_PASSWORD_READONLY
pymongo_url = $PYMONGO_URL
pymongo_user = $PYMONGO_USER_READONLY
pymongo_password = $PYMONGO_PASSWORD_READONLY
pymongo_database = $PYMONGO_DATABASE
xent_url = $PYMONGO_URL
xent_user = $PYMONGO_USER_READONLY
xent_password = $PYMONGO_PASSWORD_READONLY
xent_database = $PYMONGO_DATABASE
EOF
