import importlib
import strax
import straxen
from utilix import DB
import datetime

db = DB()

# list of contexts that gets tracked in runDB context collection
# needs to be maintained for each straxen release
context_list = ['xenonnt_led',
                'xenonnt_online',
                ]


# returns the list of dtype, hashes for a given strax context
def get_hashes(st):
    return set([(d, st.key_for('0', d).lineage_hash)
                for p in st._plugin_class_registry.values()
                for d in p.provides])


def main():
    for context in context_list:
        context_module = importlib.import_module('straxen.contexts')
        st = getattr(context_module, context)()
        hashes = get_hashes(st)
        hash_dict = {dtype: h for dtype, h in hashes}

        doc = dict(name=context,
                   date_added=datetime.datetime.utcnow(),
                   hashes=hash_dict,
                   straxen_version=straxen.__version__,
                   strax_version=strax.__version__
                   )

        # update the context collection using utilix + runDB_api
        db.update_context_collection(doc)


if __name__ == "__main__":
    main()
