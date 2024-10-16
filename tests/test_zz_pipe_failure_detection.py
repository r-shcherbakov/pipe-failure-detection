"""Final test alphabetically (zz) to catch general integration cases."""

import toml

from pipe_failure_detection import __version__


def test_version():
    """Check that PyProject and __version__ are equivalent."""
    result = toml.load('pyproject.toml')['tool']['poetry']['version']

    assert result == __version__
