import time
import pytz
import datetime

class SimpleClock:
    utc: bool
    cutoff_offset: float

    def __init__(self, utc=True, cutoff_offset=3600) -> None:
        self.utc = utc
        self.cutoff_offset = cutoff_offset

    def current_datetime(self):
        if self.utc:
            return datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        return datetime.datetime.now()

    def cutoff_datetime(self, buffer=0.):
        offset = datetime.timedelta(seconds=self.cutoff_offset+buffer)
        return self.current_datetime() + offset