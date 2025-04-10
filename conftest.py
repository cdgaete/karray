import pytest
from src.karray import settings

def pytest_addoption(parser):
    parser.addoption(
        "--data-type",
        action="store",
        default="dense",
        help="data type for array: dense or sparse"
    )

@pytest.fixture(scope="session", autouse=True)
def set_data_type(request):
    """Set the data type for all tests in the session."""
    data_type = request.config.getoption("--data-type")
    settings.data_type = data_type
    return data_type

@pytest.fixture(scope="function")
def use_dense():
    """Force the use of dense arrays for a test."""
    old_setting = settings.data_type
    settings.data_type = 'dense'
    yield
    settings.data_type = old_setting

@pytest.fixture(scope="function")
def use_sparse():
    """Force the use of sparse arrays for a test."""
    old_setting = settings.data_type
    settings.data_type = 'sparse'
    yield
    settings.data_type = old_setting
