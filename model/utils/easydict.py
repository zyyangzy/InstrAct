"""Legacy EasyDict required only to deserialize official InternVideo weights."""


class EasyDict(dict):
    """Dictionary exposing its keys as attributes.

    The class path intentionally remains ``utils.easydict.EasyDict`` because
    that fully-qualified name is embedded in the InternVideo checkpoint.
    """

    def __init__(self, values=None, **kwargs):
        values = {} if values is None else values
        values.update(kwargs)
        for key, value in values.items():
            setattr(self, key, value)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [
                self.__class__(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, values=None, **kwargs):
        values = values or {}
        values.update(kwargs)
        for key, value in values.items():
            setattr(self, key, value)

    def pop(self, key, default=None):
        if hasattr(self, key):
            delattr(self, key)
        return super().pop(key, default)
