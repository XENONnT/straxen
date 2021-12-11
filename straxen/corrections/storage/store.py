

import pandas as pd

class setionError(Exception):
    pass


class CorrectionStore:

    @staticmethod
    def construct_index(correction, *args, **kwargs):
        index = dict(zip(correction.index, args))
        index.update(kwargs)
        index = {k:v for k,v in index.items() if k in correction.index}
        return index
    
    def get(self, correction, *args, **kwargs):
        raise NotImplementedError
    
    def get_one(self, correction, *args, **kwargs):
        raise NotImplementedError

    def get_df(self, correction, *args, **kwargs):
        df = pd.DataFrame(self.get(correction, *args, **kwargs))
        return df.set_index(list(correction.index))

    def set(self, correction, **index):
        for indexer in correction.index.values():
            for field in indexer.fields:
                if field not in index:
                    raise setionError(f'{correction.__class__.__name__}\
                                         requires value for {field}')
            indexer.validate(index[field])

        if self.get(correction, **index):
            raise setionError(f'Values already defined for {index}.')
        
        correction.pre_set(**index)

        return self._set(correction, **index)

    def _set(self, correction, **index):
        raise NotImplementedError