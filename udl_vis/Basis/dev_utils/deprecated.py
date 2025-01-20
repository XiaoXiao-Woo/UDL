import warnings
from functools import wraps
from contextlib import contextmanager

# Although we use warnings.warn, it forbid to output the warning message in the console.

def deprecated(func, msg):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version. {msg}",
            DeprecationWarning,
        )
        return func(*args, **kwargs)

    return wrapper


@contextmanager
def deprecated_context(func_name, msg):
    warnings.warn(
        f"{func_name} is deprecated and will be removed in a future version. {msg}",
        DeprecationWarning,
    )
    try:
        yield
    finally:
        pass
