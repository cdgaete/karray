#!/bin/bash
set -e

echo "======================================================================"
echo "              Initializing karray Environment             "
echo "======================================================================"

# Change to project root directory
cd /workspaces/karray

# Install the package in development mode with all extras
echo "Installing package in development mode..."
pip install -e '.[all]'

# Install additional testing packages
echo "Installing testing packages..."
pip install pytest pytest-cov

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Add PYTHONPATH to bashrc if not already there
if ! grep -q "PYTHONPATH=/workspaces/karray" ~/.bashrc; then
    echo "export PYTHONPATH=/workspaces/karray:\$PYTHONPATH" >> ~/.bashrc
    echo "Added project to PYTHONPATH in ~/.bashrc"
fi

# Add testing aliases to bashrc
echo "Setting up testing aliases..."
if ! grep -q "alias run-tests" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# karray testing aliases
alias run-tests='cd /workspaces/karray && python -m pytest'
alias run-tests-verbose='cd /workspaces/karray && python -m pytest -v'
alias run-tests-cov='cd /workspaces/karray && python -m pytest --cov=src/karray'
alias run-doctests='cd /workspaces/karray && python doctests.py'
alias run-all-tests='cd /workspaces/karray && python -m pytest && python doctests.py'
alias test-dense='cd /workspaces/karray && python tests/test_dense.py'
alias test-sparse='cd /workspaces/karray && python tests/test_sparse.py'

EOF
    echo "Added testing aliases to ~/.bashrc"
fi

echo "Environment initialization complete!"
echo "======================================================================"

# Show the package version and available test commands
python -c "import karray as ka; print(f\"karray version: {ka.__version__}\")"
echo "Available test commands:"
echo "  run-tests          - Run all pytest tests"
echo "  run-tests-verbose  - Run all pytest tests with verbose output"
echo "  run-tests-cov      - Run tests with coverage report"
echo "  run-doctests       - Run doctests"
echo "  run-all-tests      - Run both pytest and doctests"
echo "  test-dense         - Run tests with dense data type"
echo "  test-sparse        - Run tests with sparse data type"
echo "======================================================================"
