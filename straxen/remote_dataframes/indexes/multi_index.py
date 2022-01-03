

import strax
import toolz
from .index import BaseIndex

export, __all__ = strax.exporter()


@export
class MultiIndex(BaseIndex):
    _indexes: list

    def __init__(self, *args, schema=None, **kwargs) -> None:
        indexes = list(args)
        for k,v in kwargs.items():
            v.name = k
            indexes.append(v)
        for i, index in enumerate(indexes):
            if index.name  in ['', 'index']:
                index.name = f'index_{i}'
        self._indexes = indexes

        if schema is not None:
            self.schema = schema

    def __set_name__(self, owner, name):
        self.schema = owner
        for v in self.indexes:
            v.schema = owner
    
    @property
    def indexes(self):
        return getattr(self, '_indexes', [])[:]

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
    
    def index_to_storage_doc(self, index):
        doc = {}
        for idx in self.indexes:
            doc.update(idx.index_to_storage_doc(index))
        return doc

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
            for _,docs in toolz.groupby(others, documents).items():
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
        docs = self.apply_query(db, query)
        docs = [dict(doc, **self.infer_index(**doc)) for doc in docs]
        docs = self.reduce(docs, index_values)
        return docs

    def __repr__(self):
        return f"MultiIndex({self.indexes})"

    def builds(self, **kwargs):
        from hypothesis import strategies as st

        @st.composite
        def strategy(draw, *index_strategies):
            return tuple(map(draw, index_strategies))

        return strategy(*[idx.builds(**kwargs.get(idx.name, {})) for idx in self.indexes])