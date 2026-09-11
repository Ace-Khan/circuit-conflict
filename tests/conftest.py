def pytest_configure(config):
    config.addinivalue_line("markers", "slow: requires loading a model")
