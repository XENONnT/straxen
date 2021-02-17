#!/bin/bash

cat > $HOME/.xenon_config <<EOF
[basic]
logging_level=debug

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
EOF
