#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
"""
Python 3.13+ compatibility module.

This module has been modernized to only support Python 3.13+ and provides
consistent interfaces and utilities for the backtrader library.
"""
from __future__ import annotations

import queue
import sys
from io import StringIO
from typing import Any, Dict, Iterator, List, Tuple, Type, TypeVar, Optional
from urllib.parse import quote as urlquote
from urllib.request import ProxyHandler, build_opener, install_opener, urlopen

# Windows registry support (optional)
try:
    import winreg
    HKEY_CURRENT_USER = winreg.HKEY_CURRENT_USER
    HKEY_LOCAL_MACHINE = winreg.HKEY_LOCAL_MACHINE
    OpenKey = winreg.OpenKey
    QueryValueEx = winreg.QueryValueEx
except ImportError:
    winreg = None  # type: ignore[assignment]
    HKEY_CURRENT_USER = None  # type: ignore[assignment]
    HKEY_LOCAL_MACHINE = None  # type: ignore[assignment]
    OpenKey = None  # type: ignore[assignment]
    QueryValueEx = None  # type: ignore[assignment]

# Windows error handling
try:
    WindowsError = WindowsError  # type: ignore[name-defined,misc]
except NameError:
    # On non-Windows systems, define a stub
    class WindowsError(OSError):
        pass

# Python 3.13+ constants
MAXINT = sys.maxsize
MININT = -sys.maxsize - 1
MAXFLOAT = sys.float_info.max
MINFLOAT = sys.float_info.min

# Type definitions for Python 3.13+
string_types = (str,)
integer_types = (int,)

# Built-in functions (no need for compatibility layer)
filter = filter
map = map
range = range
zip = zip
long = int


def cmp(a: Any, b: Any) -> int:
    """Compare two values and return -1, 0, or 1."""
    return int((a > b) - (a < b))


def bytes_func(x: str) -> bytes:
    """Convert string to bytes using UTF-8 encoding."""
    return x.encode('utf-8')


def bstr(x: Any) -> str:
    """Convert any object to string."""
    return str(x)


# Dictionary iteration helpers
_T = TypeVar('_T')
_K = TypeVar('_K')
_V = TypeVar('_V')


def iterkeys(d: Dict[_K, _V]) -> Iterator[_K]:
    """Iterate over dictionary keys."""
    return iter(d.keys())


def itervalues(d: Dict[_K, _V]) -> Iterator[_V]:
    """Iterate over dictionary values."""
    return iter(d.values())


def iteritems(d: Dict[_K, _V]) -> Iterator[Tuple[_K, _V]]:
    """Iterate over dictionary items."""
    return iter(d.items())


def keys(d: Dict[_K, _V]) -> List[_K]:
    """Get dictionary keys as a list."""
    return list(d.keys())


def values(d: Dict[_K, _V]) -> List[_V]:
    """Get dictionary values as a list."""
    return list(d.values())


def items(d: Dict[_K, _V]) -> List[Tuple[_K, _V]]:
    """Get dictionary items as a list."""
    return list(d.items())


# Modern Python 3.13+ metaclass utilities


def with_metaclass(meta: Type[type], *bases: Type[Any]) -> Type[Any]:
    """
    Create a base class with a metaclass - modernized for Python 3.13+.
    
    This function creates a temporary metaclass that will be replaced
    by the actual metaclass during class creation. This provides
    compatibility for the existing codebase while maintaining type safety.
    
    Args:
        meta: The metaclass to use
        *bases: Base classes for the new class
        
    Returns:
        A temporary class that will use the specified metaclass
    """
    class metaclass(meta):  # type: ignore[misc,valid-type]
        def __new__(cls, name: str, this_bases: Tuple[Type[Any], ...], d: Dict[str, Any]) -> Type[Any]:
            if this_bases is None:
                # First pass - creating the temporary class
                return type.__new__(cls, name, (), d)
            # Second pass - create the actual class with the target metaclass
            return meta(name, bases, d)  # type: ignore[misc]
        
        @classmethod
        def __prepare__(cls, name: str, this_bases: Tuple[Type[Any], ...]) -> Dict[str, Any]:
            if hasattr(meta, '__prepare__'):
                return meta.__prepare__(name, bases)  # type: ignore[misc,return-value]
            return {}
    
    return type.__new__(metaclass, 'temporary_class', (), {})
