def test_import():
    import sonarcam
    assert hasattr(sonarcam, "__version__")
