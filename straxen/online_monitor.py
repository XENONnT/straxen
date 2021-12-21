from strax import MongoFrontend, exporter
from straxen import uconfig

export, __all__ = exporter()

default_online_collection = 'online_monitor'


@export
def get_mongo_uri(user_key='pymongo_user',
                  pwd_key='pymongo_password',
                  url_key='pymongo_url',
                  header='RunDB'):
    user = uconfig.get(header, user_key)
    pwd = uconfig.get(header, pwd_key)
    url = uconfig.get(header, url_key)
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
        if uri is None and readonly:
            uri = get_mongo_uri()
        elif uri is None and not readonly:
            # 'not readonly' means that you want to write. Let's get
            # your admin credentials:
            uri = get_mongo_uri(header='rundb_admin',
                                user_key='mongo_rdb_username',
                                pwd_key='mongo_rdb_password',
                                url_key='mongo_rdb_url')

        if database is None:
            database = uconfig.get('RunDB', 'pymongo_database')

        super().__init__(uri=uri,
                         database=database,
                         take_only=take_only,
                         col_name=col_name,
                         *args, **kwargs)
        self.readonly = readonly
