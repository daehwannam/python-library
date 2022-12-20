
from multiprocessing import Pool


def map_with_apply_async(func, iterable, *, num_processes):
    pool = Pool(num_processes)

    async_results = []
    for elem in iterable:
        async_results.append(pool.apply_async(func, args=(elem,)))

    results = tuple(async_result.get() for async_result in async_results)
    
    pool.close()
    pool.join()

    return results
