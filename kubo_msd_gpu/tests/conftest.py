def pytest_addoption(parser):
    parser.addoption('--require-gpu',action='store_true',help='fail instead of skipping when real GPU acceptance is requested')
