from bytecode import Bytecode, Instr
'''
                         # Instr('STORE_FAST', '_res'),
                         # Instr('LOAD_FAST', self.varname[0]),
                         # Instr('STORE_FAST', '_value'),
                         # Instr('LOAD_FAST', '_res'),
                         # Instr('LOAD_FAST', '_value'),


                        # Instr('STORE_FAST', '_res'),
                        #  Instr('LOAD_FAST', '_res'),
                        # Instr('STORE_FAST', '_res'),
                        # Instr('LOAD_FAST', self.varname[0]),
                        # Instr('LOAD_FAST', '_res'),
'''
class get_local(object):
    cache = {}
    is_activate = False
    method = "append" # lambda func, *args: getattr(func, "append")(*args)

    def __init__(self, *args):
        self.varname = args

    def __call__(self, func):
        if not type(self).is_activate:
            return func

        type(self).cache[func.__qualname__] = [None] # if self.method != "append" else []
        c = Bytecode.from_code(func.__code__)
        # extra_code = []
        extra_code = [Instr('STORE_FAST', '_res'),
                      Instr('LOAD_FAST', '_res')]
        for var in self.varname:
            extra_code.extend([
                Instr('STORE_FAST', '_res'),
                Instr('LOAD_FAST', var),
                Instr('LOAD_FAST', '_res')]
            )
        extra_code.extend([
                         Instr('BUILD_LIST', 1+len(self.varname)),
                         Instr('STORE_FAST', '_result_list'),
                         Instr('LOAD_FAST', '_result_list'),
                     ])

        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            for idx, values in enumerate(res[:-1]):
                if hasattr(values, 'detach'):
                    res[idx] = values.detach().cpu()#.numpy()
                else:
                    res[idx] = values
            if self.method == "append":
                getattr(type(self).cache[func.__qualname__], self.method)(res[:-1])
            else:
                type(self).cache[func.__qualname__][0] = res[:-1]
            return res[-1]
        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True
