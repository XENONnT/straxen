#!/bin/bash

cat > $HOME/.xenon_config <<EOF
[RunDB]
rundb_api_url = $RUNDB_API_URL_READONLY
rundb_api_user = $RUNDB_API_USER_READONLY
rundb_api_password = $RUNDB_API_PASSWORD_READONLY
xent_url = $PYMONGO_URL
xent_user = $PYMONGO_USER
xent_password = $PYMONGO_PASSWORD
xent_database = $PYMONGO_DATABASE
EOF
