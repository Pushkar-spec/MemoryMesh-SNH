"""Initial verification test for MemoryMesh setup."""

import memorymesh


def test_package_metadata() -> None:
    """Verify package import and version metadata."""
    assert memorymesh.__version__ == "0.1.0"
