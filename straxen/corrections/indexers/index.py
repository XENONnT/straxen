

class Index:
    indexers: dict

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

    def build_query(self, store, **index):
        pass
    
    def apply_query(self, store, query):
        pass