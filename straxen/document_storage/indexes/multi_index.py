

import toolz
from .index import Index
from ..utils import singledispatchmethod


class MultiIndex(Index):
    indexes: list

    def __init__(self, *args, document=None, **kwargs) -> None:
        self.indexes = list(args)
        for k,v in kwargs.items():
            v.name = k
            self.indexes.append(v)
        for i, index in enumerate(self.indexes):
            if index.name  in ['', 'index']:
                index.name = f'index_{i}'

        if document is not None:
            self.document = document

    def __set_name__(self, owner, name):
        self.document = owner
        for v in self.indexes:
            v.document = owner

    @property
    def name(self):
        return self.query_fields

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

    def infer_index_value(self, **kwargs):
        return tuple(idx.infer_index_value(**kwargs) for idx in self.indexes)

    def infer_index(self, *args, **kwargs):
        index = dict(zip(self.query_fields, args))
        index.update(kwargs)
        return {idx.name: idx.infer_index_value(**index) for idx in self.indexes}
    
    def build_query(self, db, value):
        return [index.build_query(db, value) for index in self.indexes]

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

    def build_query(self, db, index_values):
        query = []
        for index in self.indexes:
            if index.name not in index_values:
                continue
            query.append(index.build_query(db, index_values[index.name]))
        return query

    def query_db(self, db, *args, **kwargs):
        index_values = self.infer_index(*args, **kwargs)
        query = self.build_query(db, index_values)
        documents = self.apply_query(db, query)
        documents = self.reduce(documents, index_values)
        return documents

    def __repr__(self):
        return f"MultiIndex({self.indexes})"