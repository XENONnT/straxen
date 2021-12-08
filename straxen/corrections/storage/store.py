

import pandas as pd


class CorrectionStore:
    @staticmethod
    def construct_index(correction, *args, **kwargs):
        index = dict(zip(correction.indices(), args))
        index.update(kwargs)
        return index
    
    def get_values(self, correction, *args, **kwargs):
        raise NotImplementedError
        
    def get_df(self, correction, *args, **kwargs):
        df = pd.DataFrame(self.get_values(correction, *args, **kwargs))
        return df.set_index(list(correction.indices()))
