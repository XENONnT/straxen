import time
import pandas as pd


class Clock:
    utc: bool
    cutoff_offset: float

    def __init__(self, utc=True, cutoff_offset=3600) -> None:
        self.utc = utc
        self.cutoff_offset = cutoff_offset

    def current_datetime(self):
        return pd.to_datetime(time.time(),
                            unit='s', utc=self.utc)

    def cutoff_datetime(self):
        return pd.to_datetime(time.time()+self.cutoff_offset,
                            unit='s', utc=self.utc)
