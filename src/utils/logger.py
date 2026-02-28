"""Structured logging utility for organized, visually pleasing console output."""

import os
import sys
from contextlib import contextmanager


# Enable ANSI escape codes on Windows
if sys.platform == 'win32':
    os.system('')


class Logger:
    """Provides structured, hierarchical console output with visual formatting."""

    _BOLD = '\033[1m'
    _DIM = '\033[2m'
    _CYAN = '\033[96m'
    _GREEN = '\033[92m'
    _YELLOW = '\033[93m'
    _RED = '\033[91m'
    _BLUE = '\033[94m'
    _RESET = '\033[0m'

    _WIDTH = 70

    def __init__(self):
        self._indent = 0
        self._colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{self._RESET}" if self._colors else text

    def _pad(self) -> str:
        return "│  " * self._indent

    def header(self, text: str):
        """Big prominent header with double-line box."""
        w = self._WIDTH
        padded = f"  {text}"
        print()
        print(self._c(self._CYAN, f"  ╔{'═' * (w - 2)}╗"))
        print(self._c(self._CYAN, f"  ║{padded:<{w - 2}}║"))
        print(self._c(self._CYAN, f"  ╚{'═' * (w - 2)}╝"))
        print()

    def section(self, text: str):
        """Opens a labeled section with a top border."""
        prefix = self._pad()
        remaining = self._WIDTH - len(prefix) - len(text) - 4
        line = '─' * max(remaining, 2)
        print(self._c(self._BLUE, f"{prefix}┌─ ") + self._c(self._BOLD, text) + self._c(self._BLUE, f" {line}"))

    def end_section(self):
        """Closes the current section with a bottom border."""
        prefix = self._pad()
        line = '─' * (self._WIDTH - len(prefix) - 1)
        print(self._c(self._BLUE, f"{prefix}└{line}"))

    @contextmanager
    def group(self, title: str):
        """Context manager that opens and closes a section automatically."""
        self.section(title)
        self._indent += 1
        try:
            yield
        finally:
            self._indent -= 1
            self.end_section()

    def info(self, message: str):
        """Standard info line."""
        prefix = self._c(self._BLUE, f"{self._pad()}│")
        print(f"{prefix}  {message}")

    def detail(self, message: str):
        """Dimmed detail/verbose line."""
        prefix = self._c(self._BLUE, f"{self._pad()}│")
        print(f"{prefix}  {self._c(self._DIM, message)}")

    def result(self, label: str, value):
        """Key-value result line with bold value."""
        prefix = self._c(self._BLUE, f"{self._pad()}│")
        print(f"{prefix}  {label}: {self._c(self._BOLD, str(value))}")

    def success(self, message: str):
        """Success line with green checkmark."""
        prefix = self._c(self._BLUE, f"{self._pad()}│")
        print(f"{prefix}  {self._c(self._GREEN, '✓')} {message}")

    def warning(self, message: str):
        """Warning line with yellow indicator."""
        prefix = self._c(self._BLUE, f"{self._pad()}│")
        print(f"{prefix}  {self._c(self._YELLOW, '⚠')} {message}")

    def error(self, message: str):
        """Error line with red indicator."""
        prefix = self._c(self._BLUE, f"{self._pad()}│")
        print(f"{prefix}  {self._c(self._RED, '✗')} {message}")

    def table(self, df):
        """Prints a DataFrame as an indented table."""
        for line in df.to_string().split('\n'):
            self.info(line)

    def blank(self):
        """Prints a continuation line."""
        print(self._c(self._BLUE, f"{self._pad()}│"))


log = Logger()
