import datetime

def tstamp():
    dt = datetime.datetime.now()
    code = '%d_%b_%Y-%H:%M'
    return dt.strftime(code)
