"""
Module: singleton

This module provides a metaclass implementation for the Singleton design pattern,
ensuring that only one instance of a class exists throughout the lifetime of an
application. It is particularly useful for scenarios where a single point of access
to shared resources or configurations is required.
"""

from typing import Dict


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances: Dict = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            # If an instance already exists, check for the "update" flag
            instance = cls._instances[cls]
            if kwargs.get("update", False):
                instance.__init__(*args, **kwargs)
        return instance
