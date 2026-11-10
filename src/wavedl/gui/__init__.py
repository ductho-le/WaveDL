"""
WaveDL Interactive Web GUI
==========================

Provides the `wavedl-gui` console script for configuring YAML experiments
and launching `wavedl-train` from a lightweight local web interface.
"""

from .app import main


__all__ = ["main"]
