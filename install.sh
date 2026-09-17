# Change current directory into project root
original_dir=$(pwd)
script_dir=$(dirname "$0")
cd "$script_dir"

# Remove old dist file, build, and install
rm -rf dist
rm -rf build
python setup_musa.py bdist_wheel
pip install --force-reinstall dist/*.whl

# Open users' original directory
cd "$original_dir"
