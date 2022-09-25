
from multiprocessing import Lock, Process, Queue, Manager, cpu_count

def _format_bytes(size: int):
    """
    Format bytes to human readable format.

    Thanks to: https://stackoverflow.com/a/70211279
    """
    power_labels = {40: "TB", 30: "GB", 20: "MB", 10: "KB"}
    for power, label in power_labels.items():
        if size >= 2 ** power:
            approx_size = size / 2 ** power
            return f"{approx_size:.1f} {label}"
    return f"{size} bytes"

def _parallelize(function=None, inputdict: dict = None, nr_workers=1, verbose=False, **kargs):
    """
    Parallelize function to run program faster.
    The queue contains tuples of keys and objects, the function must be consistent when getting data from queue.

    Args:
        function (function, optional): Function that is to be parallelized. Defaults to None.
        inputdict (dict, optional): Contains numbered keys and as value any object. Defaults to None.
        nr_workers (int, optional): Number of workers, so their tasks can run parallel. Defaults to 1.

    Returns:
        dict: Dictionary the given functions creates.
    """
    total_cpu = cpu_count()
    numb_of_tasks = len(inputdict)
    if nr_workers > total_cpu:
        nr_workers = min(numb_of_tasks,total_cpu)
    else:
        nr_workers = min(numb_of_tasks,nr_workers)
    # if verbose:
    #     logger.info(f"Workers: {nr_workers} of {total_cpu}")
    with Manager() as manager:
        dc = manager.dict()
        queue = Queue()
        for key, item in inputdict.items():
            queue.put((key, item))
        queue_lock = Lock()
        processes = {}
        for i in range(nr_workers):
            if kargs:
                processes[i] = Process(target=parallel_func,
                                        args=(
                                            dc,
                                            queue,
                                            queue_lock,
                                            function,
                                            kargs,
                                        ))
            else:
                processes[i] = Process(target=parallel_func,
                                        args=(
                                            dc,
                                            queue,
                                            queue_lock,
                                            function,
                                        ))
            processes[i].start()
        for i in range(nr_workers):
            processes[i].join()
        outputdict = dict(dc)
    return outputdict


def parallel_func(dc, queue=None, queue_lock=None, function=None, kargs={}):
    """
    #TODO DOCSTRING

    Args:
        dc ([type]): [description]
        queue ([type], optional): [description]. Defaults to None.
        queue_lock ([type], optional): [description]. Defaults to None.
        function ([type], optional): [description]. Defaults to None.
        kargs (dict, optional): [description]. Defaults to {}.

    Returns:
        [type]: [description]
    """

    while True:
        queue_lock.acquire()
        if queue.empty():
            queue_lock.release()
            return None
        key, item = queue.get()
        queue_lock.release()
        obj = function(**item, **kargs)
        dc[key] = obj

# import numpy as np
# from numba import njit
# from numba import types
# from numba.typed import Dict

# # Make key type with two 32-bit integer items.
# key_type = types.UniTuple(types.int32, 2)

# # Make array type.  Type-expression is not supported in jit functions.
# float_array = types.float64[:]

# @njit
# def foo():
#     list_out=[]
#     # Make dictionary
#     d = Dict.empty(
#         key_type=key_type, 
#         value_type=float_array,
#     )
#     # an example of how I would like to fill the dictionary
#     d[(1,1)] = np.arange(3, dtype=np.float64)
#     d[(2,2)] = np.arange((3, 6), dtype=np.float64)
#     list_out.append(d[(2,2)])
#     return list_out

# list_out = foo()
