from abc import ABC, abstractmethod


class DatasourceInterface(ABC):

    @abstractmethod
    def build_selection(self, value):
        pass


class Selection(ABC):

    @abstractmethod
    def apply_selection(self, db):
        pass