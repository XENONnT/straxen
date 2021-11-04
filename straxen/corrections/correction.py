from abc import ABC, abstractclassmethod, abstractmethod
from datetime import datetime
import pandas as pd
import typing as ty
import strax

from .model import BaseCorrectionModel, BaseIntervalCorrection
from .storage import BaseCorrectionStore, BaseIntervalStore, InsertionError


class CorrectionError(Exception):
    pass

class CorrectionInsertionError(CorrectionError):
    pass

class CorrectionReadError(CorrectionError):
    pass

class BaseCorrection(ABC):
    CORRECTIONS = {}

    name: ty.ClassVar
    Model = BaseCorrectionModel
    Storage = BaseCorrectionStore
    
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if 'Base' in cls.__name__:
            return
        if cls is BaseCorrection:
            return
        if hasattr(cls, 'name'):
            name = cls.name
        else:
            name = strax.camel_to_snake(cls.__name__)
        if name in cls.CORRECTIONS:
            raise CorrectionError(f'A correction with the name {cls.name} already exists.')
        cls.CORRECTIONS[name] = cls
        
        for superclass in cls.mro():
            if hasattr(superclass, 'Model') and not issubclass(cls.Model, superclass.Model):
                raise CorrectionError(f'Class attribute Model must be a subclass of {superclass.Model}, got {cls.Model}')
            if hasattr(superclass, 'Storage') and not issubclass(cls.Storage, superclass.Storage):
                raise CorrectionError(f'Class attribute Storage must be a subclass of {superclass.Storage}, got {cls.Storage}')    

    @classmethod
    def find(cls, storage, **kwargs):
        storage = cls.validate_storage(storage)
        docs = cls._find(storage, **kwargs)
        docs = cls.validate_documents(docs)
        return docs

    @abstractclassmethod
    def _find(cls, storage, **kwargs):
        pass
    
    @classmethod
    def validate_documents(cls, docs):
        validated = []
        for doc in docs:
            if isinstance(doc, cls.Model):
                 validated.append(doc)
            if isinstance(doc, ty.Mapping):
                validated.append(cls.Model(**doc))
            else:
                raise InsertionError('Documents must be of type {cls.Model} or dict.')
        return validated

    @classmethod
    def validate_storage(cls, storage):
        if not isinstance(storage, cls.Storage):
            raise TypeError('Storage class is not suitable \
                 for this type of correction.')
        return storage

    @classmethod
    def insert(cls, storage, docs=None, **kwargs):
        storage = cls.validate_storage(storage)
        if docs is None:
            docs = kwargs

        if not isinstance(docs, ty.Iterable):
            docs = [docs]
        docs = cls.validate_documents(docs)
        return cls._insert(storage, docs)

    @abstractclassmethod
    def _insert(cls, storage, docs):
        pass

    def update(cls, storage, docs=None, **kwargs):
        storage = cls.validate_storage(storage)
        if docs is None:
            docs = kwargs
        if not isinstance(docs, ty.Iterable):
            docs = [docs]
        docs = cls.validate_documents(docs)
        return cls._update(storage ,docs)

    @abstractclassmethod
    def _update(cls, storage, docs):
        pass

    
class BaseIntervalCorrection(BaseCorrection):
    Storage = BaseIntervalCorrection
    
    @abstractclassmethod
    def overlaps(cls, storage,  begin, end=None, **kwargs):
        if not isinstance(storage,  cls.Storage):
            raise TypeError('Storage class is not suitable \
                 for this type of correction.')
        return storage.overlaps(begin, end=end, **kwargs)

    @classmethod
    def _find(cls, storage, **kwargs):
        pass

    @classmethod
    def _insert(cls, storage, docs):
        pass

    @classmethod
    def _update(cls, storage, docs):
        pass

class BaseInterpolatingCorrection(BaseCorrection):

    @abstractclassmethod
    def interpolate(cls, storage, index, **kwargs):
        storage = cls.validate_storage(storage)
        if isinstance(index, cls.Model):
            index = index.index

    @classmethod
    def _find(cls, index, **kwargs):
        pass

class BaseCorrectionClient:
    def __init__(self, correction, storage) -> None:
        self.correction = correction
        self.storage = storage

    def value_at(self, index, **kwargs):
        overlaps = self.overlaps(index, **kwargs)
        if overlaps:
            return overlaps[-1].get_value()
        
    def get_values(self, **kwargs):
        return [c.get_value() for c in self.find(**kwargs)]
    
    def get_value(self, **kwargs):
        return self.find_one().get_value()