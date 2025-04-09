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

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install



# Add PYTHONPATH to bashrc if not already there
if ! grep -q "PYTHONPATH=/workspaces/karray" ~/.bashrc; then
    echo "export PYTHONPATH=/workspaces/karray:\$PYTHONPATH" >> ~/.bashrc
    echo "Added project to PYTHONPATH in ~/.bashrc"
fi

echo "Environment initialization complete!"
echo "======================================================================"

# Show the package version
python -c "import karray as ka; print(f\"karray version: {ka.__version__}\")"
