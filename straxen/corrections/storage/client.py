
from ..correction import BaseCorrection
from .store import CorrectionStore

class CorrectionClient:
    correction: type
    store: CorrectionStore
    
    def __init__(self, correction, store):
        self.correction = correction
        self.store = store
        
    def get(self, *args, **kwargs):
        docs = self.store.get(self.correction, *args, **kwargs)
        if len(docs)==1:
            return docs[0]
        return docs
    
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )
        docs = self.get(*index)
        nfields = len(self.correction.index.query_fields)
        if len(index)>nfields:
            for k in index[nfields:]:
                docs = docs[k]
        return docs

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        if not isinstance(value, dict):
            value = {'value': value}
        self.set(*key, **value)

    def set(self, *args, **kwargs):
        doc = self.correction(**kwargs)
        index = self.store.construct_index(self.correction, *args, **kwargs)
        return self.store.set(doc, **index)

class CorrectionsClient:
    store: CorrectionStore
    
    def __init__(self, store):
        self.store = store
    
    @classmethod
    def default(cls):
        from .mongo_store import MongoCorrectionStore
        store = MongoCorrectionStore('cmt2')
        return cls(store)

    @property
    def corrections(self):
        return BaseCorrection.correction_classes()
    
    @property
    def correction_names(self):
        return list(self.corrections)

    def __getitem__(self, key):
        if isinstance(key, tuple) and key[0] in self.corrections:
            correction = self.corrections[key[0]]
            return CorrectionClient(correction, self.store)[key[1:]]
        
        if key in self.corrections:
            correction = self.corrections[key]
            return CorrectionClient(correction, self.store)
        raise KeyError(key)
        
    def __dir__(self):
        return super().__dir__() + list(self.corrections)
    
    def __getattr__(self, name):
        if name in self.corrections:
            return self[name]
        raise AttributeError(name)

    def get(self, correction_name, *args, **kwargs):
        correction = self.corrections[correction_name]
        return CorrectionClient(correction, self.store).get(*args, **kwargs)

    def set(self, correction_name, **kwargs):
        correction = self.corrections[correction_name]
        return CorrectionClient(correction, self.store).set(**kwargs)
