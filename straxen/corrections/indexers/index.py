
import toolz
import pymongo
from itertools import product
from dask.utils import Dispatch


class Index:
    indexers: dict
    apply_selection = Dispatch('apply_selection')

    def __init__(self, **indexers) -> None:
        for k,v in indexers.items():
            v.name = k
        self.indexers = indexers

    def __set_name__(self, owner, name):
        self.correction = owner
        self.name = name

    @property
    def query_fields(self):
        fields = ()
        for indexer in self.indexers.values():
            fields += indexer.query_fields
        return fields

    @property
    def store_fields(self):
        fields = ()
        for indexer in self.indexers.values():
            fields += indexer.store_fields
        return fields

    def keys(self):
        return self.indexers.keys()

    def values(self):
        return self.indexers.values()

    def items(self):
        return self.indexers.items()

    def __iter__(self):
        yield from self.indexers

    def __getitem__(self, key):
        return self.indexers[key]

    def __len__(self):
        return len(self.indexers)

    def build_index(self, *args, **kwargs):
        index = dict(zip(self.query_fields, args))
        index.update(kwargs)
        index = {k:v for k,v in index.items() if k in self.query_fields}
        return index

    def query_db(self, db, **index):
        db = self.apply_selection(db, self.correction)
        for name, indexer in self.indexers.items():
            if name not in index:
                continue
            db = indexer.query_db(db, name, index[name])
        documents = []
        for d in list(db):
            if isinstance(d, dict):
                documents.append(d)
            else:
                documents.extend(list(d))
        return documents

    def process(self, documents, **index):
        for name, indexer in self.indexers.items():
            if name not in index:
                continue
            documents = indexer.process(name, 
                                        index[name],
                                        documents)
        return documents

    def reduce(self, documents, **index):
        if not documents:
            return documents
        keys = set([k for indexer in self.indexers.values() for k in indexer.store_fields])
        keys = keys.intersection(documents[0])
        for name, indexer in self.indexers.items():
            if name not in index:
                continue
            others = [k for k in keys if k not in indexer.store_fields]
            if not others:
                continue
            reduced_documents = []
            for idx_values,docs in toolz.groupby(others, documents).items():
                idx = dict(zip(others, idx_values))
                idx[name] = index[name]
                reduced  = indexer.reduce(name, docs, **idx)
                reduced_documents.extend(reduced)
            documents = reduced_documents
        return documents

    def find(self, db, *args, **kwargs):
        index = self.build_index(*args, **kwargs)
        documents = self.query_db(db, **index)
        documents = self.process(documents, **index)
        documents = self.reduce(documents, **index)
        return documents

    def set(self, db, doc):
        pass

@Index.apply_selection.register(pymongo.database.Database)
def mongo_db(db, correction):
    return db[correction.name].find(projection={'_id': 0})

@Index.apply_selection.register(pymongo.collection.Collection)
def mongo_collection(db, correction):
    return db.find({'name': correction.name}, projection={'_id': 0})
