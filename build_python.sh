mkdir -p build/python
cd core/python
python setup.py egg_info --egg-base ../../build/python sdist -d ../../build/python bdist_wheel -d ../../build/python
python setup.py clean