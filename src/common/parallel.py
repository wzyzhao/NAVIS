"""Ordered task execution with bounded submission and exception propagation."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor


def ordered_map[T, R](function: Callable[[T], R], items: Iterable[T], *, workers: int = 1) -> Iterator[R]:
    """Map independent tasks in input order, keeping at most twice workers in flight.

    Use threads for I/O or native kernels that release the GIL. CPU-bound Python
    work should run in subprocesses inside the supplied function. Exceptions are
    raised at the corresponding input position; pending jobs are cancelled.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        yield from map(function, items)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video") as executor:
        pending: deque[Future[R]] = deque()
        try:
            for _ in range(2 * workers):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                pending.append(executor.submit(function, item))
            while pending:
                yield pending.popleft().result()
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.append(executor.submit(function, item))
        finally:
            for future in pending:
                future.cancel()
