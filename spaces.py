import functools

class GPU:
    def __init__(self, duration=None):
        self.duration = duration

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
