

import pandas as pd

class InsertionError(Exception):
    pass


class CorrectionStore:

    @staticmethod
    def construct_index(correction, *args, **kwargs):
        index = dict(zip(correction.indices(), args))
        index.update(kwargs)
        index = {k:v for k,v in index.items() if k in correction.indices()}
        return index
    
    def get_values(self, correction, *args, **kwargs):
        raise NotImplementedError
    
    def get_value(self, correction, *args, **kwargs):
        raise NotImplementedError

    def get_df(self, correction, *args, **kwargs):
        df = pd.DataFrame(self.get_values(correction, *args, **kwargs))
        return df.set_index(list(correction.indices()))

    def insert(self, correction, **index):
        for indexer in correction.indices().values():
            for field in indexer.fields:
                if field not in index:
                    raise InsertionError(f'{correction.__class__.__name__}\
                                         requires value for {field}')
            indexer.validate(index[field])

        if self.get_values(correction, **index):
            raise InsertionError(f'Values already defined for {index}.')
        
        correction.pre_insert(**index)

        return self._insert(correction, **index)

    def _insert(self, correction, **index):
        raise NotImplementedError