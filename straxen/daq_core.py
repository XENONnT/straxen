"""
Core functions of the DAQ, mostly used in straxen/bin
"""

import pymongo
import utilix
from utilix import uconfig
import straxen
from datetime import datetime, timedelta
import pytz

ceph_folder = '/live_data/xenonnt/'
output_folder = '/data/xenonnt_processed/'
pre_folder = '/data/pre_processed/'
non_registered_folder = '/data/xenonnt_unregistered/'

class DataBases:
    def __init__(self, production=False):
        self.production = production
        # DAQ database
        daq_db_name = 'daq'
        daq_uri = straxen.get_mongo_uri(header='rundb_admin',
                                        user_key='mongo_daq_username',
                                        pwd_key='mongo_daq_password',
                                        url_key='mongo_daq_url')
        daq_client = pymongo.MongoClient(daq_uri)
        self.daq_db = daq_client[daq_db_name]
        self.bs_coll = self.daq_db['eb_monitor']
        self.ag_stat_coll = self.daq_db['aggregate_status']
        self.log_coll = self.daq_db['log']

        # Runs database
        run_dbname = straxen.uconfig.get('rundb_admin', 'mongo_rdb_database')
        run_collname = 'runs'
        if production:
            self.run_db = self.get_admin_client()[run_dbname]
        else:
            # Please note, this is a read only account on the rundb
            run_uri = straxen.get_mongo_uri()
            run_client = pymongo.MongoClient(run_uri)
            self.run_db = run_client[run_dbname]
        self.run_coll = self.run_db[run_collname]

    @staticmethod
    def get_admin_client():
        # We want admin access to start writing data!
        mongo_url = uconfig.get('rundb_admin', 'mongo_rdb_url')
        mongo_user = uconfig.get('rundb_admin', 'mongo_rdb_username')
        mongo_password = uconfig.get('rundb_admin', 'mongo_rdb_password')
        mongo_database = uconfig.get('rundb_admin', 'mongo_rdb_database')

        collection = utilix.rundb.xent_collection(
            url=mongo_url, user= mongo_user, password=mongo_password, database=mongo_database)

        # Do not delete the client!
        return collection.database.client

    def log_warning(self,
                    message,
                    priority='warning',
                    run_id=None,
                    production=True,
                    user='daq_process',
    ):
        """Report a warning to the terminal (using the logging module)
        and the DAQ log DB.
        :param message: insert string into log_coll
        :param priority: severity of warning. Can be:
            info: 1,
            warning: 2,
            <any other valid python logging level, e.g. error or fatal>: 3
        :param run_id: optional run id.
        """
        if not production:
            return

        # Log according to redax rules
        # https://github.com/coderdj/redax/blob/master/MongoLog.hh#L22
        warning_message = {
            'message': message,
            'user': user,
            'priority': dict(debug=0, info=1, warning=2, error=3, fatal=4,).get(priority.lower(), 3)}
        if run_id is not None:
            warning_message.update({'runid': int(run_id)})
        self.log_coll.insert_one(warning_message)


def now(plus=0):
    """Now in utc time"""
    return datetime.now(pytz.utc) + timedelta(seconds=plus)