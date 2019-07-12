import functools
import sys
from copy import deepcopy

# assert 'joblib' not in sys.modules

# import pickle


# class TunedUnframer(pickle._Unframer):
#     def read_one(self):
#         if self.current_frame:
#             data = self.current_frame.read(1)
#             if not data:
#                 self.current_frame = None
#                 return self.file_read(1)
#             if len(data) == 0:
#                 raise pickle.UnpicklingError(
#                     "pickle exhausted before end of frame")
#             return data
#         else:
#             return self.file_read(1)


# class TunedUnpickler(pickle._Unpickler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._unframer = pickle._Unframer(self._file_read, self._file_readline)
#         self.read = self._unframer.read
#         self.readline = self._unframer.readline
#
#     def load(self):
#         """Read a pickled object representation from the open file.
#
#         Return the reconstituted object hierarchy specified in the file.
#         """
#         # Check whether Unpickler was initialized correctly. This is
#         # only needed to mimic the behavior of _pickle.Unpickler.dump().
#         if not hasattr(self, "_file_read"):
#             raise pickle.UnpicklingError("Unpickler.__init__() was not called by "
#                                          "%s.__init__()" % (self.__class__.__name__,))
#         self.metastack = []
#         self.stack = []
#         self.append = self.stack.append
#         self.proto = 0
#         dispatch = self.dispatch
#         # read_one = self.read_one
#         file_read = self._unframer.file_read
#         try:
#             while True:
#                 # key = read_one()
#                 if self._unframer.current_frame:
#                     data = self._unframer.current_frame.read(1)
#                     if not data:
#                         self._unframer.current_frame = None
#                         key = file_read(1)
#                     elif len(data) == 0:
#                         raise pickle.UnpicklingError(
#                             "pickle exhausted before end of frame")
#                     else:
#                         key = data
#                 else:
#                     key = file_read(1)
#
#                 if not key:
#                     raise EOFError
#                 assert isinstance(key, pickle.bytes_types)
#                 dispatch[key[0]](self)
#         except pickle._Stop as stopinst:
#             return stopinst.value
#
#
# pickle._Unpickler = TunedUnpickler

import joblib
# noinspection PyProtectedMember
from joblib._compat import PY3_OR_LATER
# noinspection PyProtectedMember
from joblib.func_inspect import _clean_win_chars
# noinspection PyProtectedMember
from joblib.memory import MemorizedFunc, _FUNCTION_HASHES, NotMemorizedFunc, Memory

_FUNC_NAMES = {}


class TunedMemory(Memory):
    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return functools.partial(self.cache, ignore=ignore,
                                     verbose=verbose, mmap_mode=mmap_mode)
        if self.store_backend is None:
            return NotMemorizedFunc(func)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, TunedMemorizedFunc):
            func = func.func
        return TunedMemorizedFunc(func, location=self.store_backend,
                                  backend=self.backend,
                                  ignore=ignore, mmap_mode=mmap_mode,
                                  compress=self.compress,
                                  verbose=verbose, timestamp=self.timestamp)


class TunedMemorizedFunc(MemorizedFunc):
    def __call__(self, *args, **kwargs):
        original_result = MemorizedFunc.__call__(self, *args, **kwargs)

        if self.func not in _FUNCTION_HASHES:
            # Also store in the in-memory store of function hashes
            if PY3_OR_LATER:
                is_named_callable = (hasattr(self.func, '__name__') and
                                     self.func.__name__ != '<lambda>')
            else:
                is_named_callable = (hasattr(self.func, 'func_name') and
                                     self.func.func_name != '<lambda>')
            if is_named_callable:
                # Don't do this for lambda functions or strange callable
                # objects, as it ends up being too fragile
                func_hash = self._hash_func()
                try:
                    _FUNCTION_HASHES[self.func] = func_hash
                except TypeError:
                    # Some callable are not hashable
                    pass

        return original_result


old_get_func_name = joblib.func_inspect.get_func_name


def tuned_get_func_name(func, resolv_alias=True, win_characters=True):
    if (func, resolv_alias, win_characters) not in _FUNC_NAMES:
        _FUNC_NAMES[(func, resolv_alias, win_characters)] = old_get_func_name(func, resolv_alias, win_characters)

        if len(_FUNC_NAMES) > 1000:
            # keep cache small and fast
            for idx, k in enumerate(_FUNC_NAMES.keys()):
                if idx % 2:
                    del _FUNC_NAMES[k]
        # print('cache size ', len(_FUNC_NAMES))

    return deepcopy(_FUNC_NAMES[(func, resolv_alias, win_characters)])


joblib.func_inspect.get_func_name = tuned_get_func_name
joblib.memory.get_func_name = tuned_get_func_name

# class TunedUnframer(_Unframer):
#     def read_one(self):
#         if self.current_frame:
#             data = self.current_frame.read(1)
#             if not data:
#                 self.current_frame = None
#                 return self.file_read(1)
#             if len(data) == 0:
#                 raise UnpicklingError(
#                     "pickle exhausted before end of frame")
#             return data
#         else:
#             return self.file_read(1)
