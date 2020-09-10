"""Return corrections from corrections DB
"""
from datetime import datetime
from datetime import timezone
import pytz
import os
from bson.son import SON
import pandas as pd
import pymongo
import numpy as np

import strax

export, __all__ = strax.exporter()


@export
class CmtServices():
    """
    A class that returns corrections
    Corrections are set of parameters to be applied in the analysis
    stage to remove detector effects. Information on the strax implementation
    can be found at https://github.com/AxFoundation/strax/blob/master/strax/corrections.py
    """
    def __init__(self):

        self.interface = strax.CorrectionsInterface(
                host='xenon1t-daq.lngs.infn.it',
                username='corrections',
                password=os.environ['CORRECTIONS_PASSWORD'],
                database_name='corrections')

        """
        :param host: DB host
        :param username: DB username
        :param password: DB password
        :param database_name: DB name
        """

        print('Calling Correction Managment Tool')

    def get_elife(self, run_id, global_version='v1', xenon1t=False):
        """
        Smart logic to return electron lifetime correction
        :param run_id: run id from runDB
        :param global_version: global version
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: electron lifetime correction value
        """
        when = CmtServices.get_time(run_id, xenon1t)
        if xenon1t:
            df_global = self.interface.read('global_xenon1t')
        else:
            df_global = self.interface.read('global')
        try:
            for correction, version in df_global.iloc[-1][global_version].items():
                if 'elife' in correction:
                    df = self.interface.read(correction)
                    df = self.interface.interpolate(df, when)
        except KeyError:
            raise ValueError(f'Global version {global_version} not found')

        return df.loc[df.index == when, global_version].values[0]

    def get_pmt_gains(self, run_id, global_version='v1', xenon1t=False):
        """
        Smart logic to return pmt gains.
        :param run_id: run id from runDB
        :param global_version: global version
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :retrun: array of pmt gains
        """
        when = CmtServices.get_time(run_id, xenon1t)
        if xenon1t:
            df_global = self.interface.read('global_xenon1t')
        else:
            df_global = self.interface.read('global')
        try:
            # equivalent to 'to_pe' in gains_model
            gains = []
            for correction, version in df_global.iloc[-1][global_version].items():
                if 'pmt' in correction:
                    df = self.interface.read(correction)
                    df = self.interface.interpolate(df, when)
                    gains.append(df.loc[df.index == when, version].values[0])
                pmt_gains = np.asarray(gains, dtype=np.float32)

        except KeyError:
            raise ValueError(f'Global version {global_version} not found')

        return pmt_gains

    def get_lce(self, run_id, s, position, global_version='v1', xenon1t=False):
        """
        Smart logic to return light collection eff map values.
        :param run_id: run id from runDB
        :param s: S1 map or S2 map
        :param postion: event position
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        """
        raise NotImplementedError

    def get_fdc(slef, run_id, position, global_version='v1', xenon1t=False):
        """
        Smart logic to return field distortion map values.
        :param run_id: run id from runDB
        :param postion: event position
        :param xenon1t: boolean xenon1t data=False/True
        """
        raise NotImplementedError

    @staticmethod
    def get_time(run_id, xenon1t=False):
        """
        Smart logic to return start time from runsDB
        :param run_id: run id from runDB
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :retrun: run start time
        """

        collection = pymongo.MongoClient(
                'mongodb://pax:@xenon1t-daq.lngs.infn.it:27017/run',
                password=os.environ['DB_password'])['xenonnt']['run/runs']

        pipeline = [
                {'$match':
                    {"number": run_id, "detector": "tpc", "end":
                        {"$exists": True}}},
                {"$project": {'time': '$start', 'number': 1, '_id': 0}},
                {"$sort": SON([("time", 1)])}]

        if xenon1t:
            collection = pymongo.MongoClient(
                    'mongodb://pax:@xenon1t-daq.lngs.infn.it:27017/run',
                    password=os.environ['DB_password'])['run']['runs_new']

            pipeline = [
                    {'$match':
                        {"name": run_id, "detector": "tpc", "end":
                            {"$exists": True}}},
                    {"$project": {'time': '$start', 'name': 1, '_id': 0}},
                    {"$sort": SON([("time", 1)])}]

        # to save it in datetime format
        time = datetime.now(tz=timezone.utc)
        for t in collection.aggregate(pipeline):
            time = t['time']
        return time.replace(tzinfo=pytz.utc)
