

import toolz
from .index import Index
from ..utils import singledispatchmethod


class MultiIndex(Index):
    indexes: list

    def __init__(self, correction=None, **indexes) -> None:
        for k,v in indexes.items():
            v.name = k
        self.indexes = list(indexes.values())
        if correction is not None:
            self.correction = correction

    def __set_name__(self, owner, name):
        self.correction = owner
        self.name = name
        for v in self.indexes:
            v.correction = owner

    @property
    def query_fields(self):
        fields = ()
        for indexer in self.indexes:
            fields += indexer.query_fields
        return fields

    @property
    def store_fields(self):
        fields = ()
        for indexer in self.indexes:
            fields += indexer.store_fields
        return fields

    def build_query(self, db, value):
        return [index.build_query(db, value) for index in self.indexes]

    def build_index(self, *args, **kwargs):
        index = dict(zip(self.query_fields, args))
        index.update(kwargs)
        index = {k:v for k,v in index.items() if k in self.query_fields}
        return index

    def reduce(self, documents, index_values):
        if not documents:
            return documents
        keys = set([k for index in self.indexes for k in index.store_fields])
        keys = keys.intersection(documents[0])
        for index in self.indexes:
            if index.name not in index_values:
                continue
            others = [k for k in keys if k not in index.store_fields]
            if not others:
                continue
            reduced_documents = []
            for idx_values,docs in toolz.groupby(others, documents).items():
                idx = dict(zip(others, idx_values))
                value = index_values[index.name]
                reduced  = index.reduce(docs, value)
                reduced_documents.extend(reduced)
            documents = reduced_documents
        return documents

    def query_db(self, db, *args, **kwargs):
        index_values = self.build_index(*args, **kwargs)
        query = []
        for index in self.indexes:
            if index.name not in index_values:
                continue
            query.append(index.build_query(db, index_values[index.name]))
        documents = self.apply_query(db, query)
        documents = self.reduce(documents, index_values)
        return documents