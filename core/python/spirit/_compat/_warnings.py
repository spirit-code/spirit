__all__ = ("deprecated",)


try:
    from warnings import deprecated
except ImportError:

    class deprecated:
        def __init__(self, message, /, *, category=DeprecationWarning, stacklevel=1):
            if not isinstance(message, str):
                raise TypeError(
                    f"Expected an object of type str for 'message', not {type(message).__name__!r}"
                )
            self.message = message
            self.category = category
            self.stacklevel = stacklevel

        def __call__(self, arg, /):
            # Make sure the inner functions created below don't
            # retain a reference to self.
            msg = self.message
            category = self.category
            stacklevel = self.stacklevel
            if category is None:
                arg.__deprecated__ = msg
                return arg
            elif callable(arg):
                import functools
                import warnings

                @functools.wraps(arg)
                def wrapper(*args, **kwargs):
                    warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
                    return arg(*args, **kwargs)

                arg.__deprecated__ = wrapper.__deprecated__ = msg
                return wrapper
            else:
                raise TypeError(
                    "@deprecated decorator with non-None category must be applied to "
                    f"a callable, not {arg!r}"
                )
