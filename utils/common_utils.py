import time

def current_timestamp():
    localtime = time.asctime( time.localtime(time.time()))
    return time.strftime("%Y%m%d%H%M%S", time.localtime())