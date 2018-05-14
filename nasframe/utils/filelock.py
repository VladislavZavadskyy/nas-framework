import os
import time

from os.path import exists
from nasframe.utils import logger


class FileLock:
    """
    A file locking mechanism.

    Args:
        path: path to attempt to acquire lock for.
        delay: delay between attempts.
    """
    def __init__(self, path, delay=.1):
        self.is_locked = False
        self.lockfile = path + ".lock"
        self.delay = delay

    @property
    def available(self):
        """
        Indicates whether the path is available to be locked.
        """
        return not exists(self.lockfile)

    def acquire(self, blocking=True):
        """
        Tries to acquire the lock.

        Args:
            blocking: if the lock is unavailable and `blocking` is False, will return False,
                         otherwise will block the next statement execution until the lock is available.
        """
        logger.debug(f'Attempting to acquire {self.lockfile}')
        while True:
            try:
                fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                with os.fdopen(fd, 'w') as f:
                    f.write('\n')
                break
            except FileExistsError:
                if not blocking:
                    return False
                time.sleep(self.delay)
        self.is_locked = True
        logger.debug(f'Acquired {self.lockfile}')
        return True

    def release(self):
        """
        Releases the lock.
        """
        if self.is_locked:
            self.is_locked = False
            os.remove(self.lockfile)
        else:
            raise RuntimeError('The file lock is not owned by this instance.')

    def purge(self):
        """
        Removes the lock file, even if it wasn't created by the current instance.

        Returns:
            True if the file existed and was successfully removed, False otherwise.
        """
        if exists(self.lockfile):
            self.is_locked = False
            os.remove(self.lockfile)
            return True
        return False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def __del__(self):
        if self.is_locked:
            self.release()