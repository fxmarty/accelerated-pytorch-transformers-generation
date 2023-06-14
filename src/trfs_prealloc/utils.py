import functools

def recurse_getattr(obj, attr: str):
    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))

def recurse_hasattr(obj, attr):
    try: left, right = attr.split('.', 1)
    except: return hasattr(obj, attr)
    return recurse_hasattr(getattr(obj, left), right)

def recurse_setattr(module, name, value):
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)

def recurse_delattr(obj, attr: str):
    if "." not in attr:
        delattr(obj, attr)
    else:
        root = ".".join(attr.split(".")[:-1])
        end = attr.split(".")[-1]
        delattr(recurse_getattr(obj, root), end)