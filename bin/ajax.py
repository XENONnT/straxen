#!/usr/bin/env python
"""
AJAX: XENONnT
Aggregate Junking Ancient Xenondata
cleaning tool to remove old data from eventbuilders
=============================================
Joran Angevaare, 2020

"""
__version__ = '0.0.0'

import argparse
from datetime import datetime, timedelta
import logging
import os
import socket
import shutil
import pymongo
import pytz


def now(plus=0):
    return datetime.now(pytz.utc) + timedelta(seconds=plus)


def _delete_data(rd, path, data_type, test= True):
    """After completing the processing and updating the runsDB, remove the
    live_data"""

    if data_type == 'live' and not args.delete_live and args.production:
        raise ValueError('Unsafe operation. Trying to delete live data!')
    if os.path.exists(path):
        log.info(f'Deleting data at {path}')
        if not test:
            shutil.rmtree(path)
    log.info(f'deleting {path} finished')
    # Remove the data location from the rundoc and append it to the 'deleted_data' entries
    if not os.path.exists(path):
        log.info('changing data field in rundoc')
        for ddoc in rd['data']:
            if ddoc['type'] == data_type:
                break
        for k in ddoc.copy().keys():
            if k in ['location', 'meta', 'protocol']:
                ddoc.pop(k)

        ddoc.update({'at': now(), 'by': hostname})
        log.info(f'update with {ddoc}')
        if not test:
            log.info('update test')
            run_coll.update_one({'_id': rd['_id']},
                                {"$addToSet": {'deleted_data': ddoc},
                                 "$pull": {"data":
                                               {"type": data_type,
                                                "host": {'$in':['daq', hostname]}}}})
    elif not test:
        raise ValueError(f"Something went wrong we wanted to delete {path}!")


if __name__ == '__main__':
    print(f'---\n ajax version {__version__}\n---')
    logging.basicConfig(level=logging.INFO,
                        format='%(relativeCreated)6d %(threadName)s %(name)s %(message)s')
    log = logging.getLogger()

    parser = argparse.ArgumentParser(
        description="XENONnT cleaning manager")
    parser.add_argument('--force', action='store_true',
                        help="Forcefully remove stuff from this host")
    parser.add_argument('--test', action='store_true',
                        help="Forcefully remove stuff from this host")

    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--number', type=int, metavar='NUMBER',
                         help="Process a single run, regardless of its status.")
    actions.add_argument('--all', action='store_true',
                         help="remove all runs from this eb that are transferred")

    args = parser.parse_args()
    run_id = '%06d' % args.number

    hostname = socket.getfqdn()

    # The event builders write to different directories on the respective machines.
    eb_directories = {
        'eb0.xenon.local': '/data2/xenonnt_processed/',
        'eb1.xenon.local': '/data1/xenonnt_processed/',
        'eb2.xenon.local': '/nfs/eb0_data1/xenonnt_processed/',
        'eb3.xenon.local': '/data/xenonnt_processed/',
        'eb4.xenon.local': '/data/xenonnt_processed/',
        'eb5.xenon.local': '/data/xenonnt_processed/',
    }

    # Set the output folder
    output_folder = eb_directories[hostname]
    if os.access(output_folder, os.W_OK) is not True:
        raise IOError(f'No writing access to {output_folder}')

    # Runs database
    run_dbname = 'xenonnt'
    run_collname = 'runs'

    # Please note, this is a read only account on the rundb
    run_db_username = os.environ['MONGO_RDB_USERNAME']
    run_db_password = os.environ['MONGO_RDB_PASSWORD']
    run_client = pymongo.MongoClient(
        f"mongodb://{run_db_username}:{run_db_password}@xenon1t-daq:27017,old-gw:27017/admin")
    run_db = run_client[run_dbname]
    run_coll = run_db[run_collname]

    # Query the database to remove data
    rd = run_coll.find_one({
        'number': args.number,
        'data.host': hostname}
        )
    if not rd:
        log.warning(f'No data for {run_id} found! Double checking on the disk!')
        deleted_data = False
        for folder in os.listdir(output_folder):
            if run_id in folder:
                log.info(f'Cleaning {output_folder + folder}')
                if not args.test:
                     shutil.rmtree(output_folder + folder)
                deleted_data = True
        if not deleted_data:
            raise FileNotFoundError(f'No data registered on {hostname} for {args.number}')
    else:
        have_live_data = False
        for dd in rd['data']:
            if dd['type'] == 'live' and dd['location'] != 'deleted':
                have_live_data = True
                break
        for ddoc in rd['data']:
            if 'host' in ddoc and ddoc['host'] == hostname:
                loc = ddoc['location']
                if not args.force and not have_live_data and 'raw_records' in ddoc['type']:
                    log.info(f'prevent {loc} from being deleted. The live_data has already'
                             f' been removed')
                elif os.path.exists(loc):
                    log.info(f'delete data at {loc}')
                    _delete_data(rd, loc, ddoc['type'], test= args.test)
                else:
                    loc = loc + '_temp'
                    log.info(f'delete data at {loc}')
                    _delete_data(rd, loc, ddoc['type'], test= args.test)
    log.info(f'Ajax did {run_id}, bye bye')