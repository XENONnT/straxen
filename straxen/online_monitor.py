from strax import MongoFrontend, exporter
from straxen import uconfig

export, __all__ = exporter()

default_online_collection = 'online_monitor'


def get_mongo_url(user_key='mongo_user',
                  pwd_key='mongo_password',
                  url_key='pymongo_url',
                  header='RunDB'):
    user=uconfig.get(header, user_key)
    pwd=uconfig.get(header, pwd_key)
    url=uconfig.get(header, url_key)
    return f"mongodb://{user}:{pwd}@{url}"


@export
class OnlineMonitor(MongoFrontend):
    """
    Online monitor Frontend for Saving data temporarily to the
    database
    """

    def __init__(self,
                 uri=None,
                 take_only=None,
                 database=None,
                 col_name=default_online_collection,
                 readonly=True,
                 *args, **kwargs):
        if take_only is None:
            raise ValueError(f'Specify which data_types to accept! Otherwise '
                             f'the DataBase will be overloaded')
        if uri is None:
            uri = get_mongo_url()

        if database is None:
            database = uconfig.get('RunDB', 'pymongo_database')

        super().__init__(uri=uri,
                         database=database,
                         take_only=take_only,
                         col_name=col_name,
                         *args, **kwargs)
        self.readonly = readonly
