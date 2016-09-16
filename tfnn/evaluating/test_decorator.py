import time


class Decorator(object):

    @staticmethod
    def time_dec(func):
        def wrapper(*arg, **kwargs):
            t = time.time()
            res = func(*arg, **kwargs)
            print(func.__name__, round(time.time()-t, 4))
            return res
        return wrapper