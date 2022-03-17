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
            now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.now()
        return self.normalize_tz(now)

    def cutoff_datetime(self, buffer=0.):
        offset = datetime.timedelta(seconds=self.cutoff_offset+buffer)
        return self.current_datetime() + offset

    def normalize_tz(self, dt: datetime.datetime ) -> datetime.datetime:
        if dt.tzinfo is not None:
            if dt.tzinfo.utcoffset(dt) is not None:
                dt = dt.astimezone(pytz.utc)
            dt = dt.replace(tzinfo=None)
        dt = dt.replace(microsecond=int(dt.microsecond/1000)*1000)
        return dt

    def after_cutoff(self, dt, buffer=0.):
        cutoff = self.cutoff_datetime(buffer)
        dt = self.normalize_tz(dt)
        return dt > cutoff 