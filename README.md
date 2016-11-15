# Activity Recognition
Conditional random fields (CRFs) and other discriminative models for activity recognition on home environments using binary sensors

#Dependencies
- Python 2.7+
- SciPy (scikitlearn and numpy)

#Quickstart
1. Run `script.py` - this will create training and testing data files in the folder `CRF++-0.58`.
2. cd to `CRF++-0.58` and run training (`crf_learn template training.txt model`) and testing (`crf_test -m model testing.txt >> results.txt`).
3. Copy results back into parent directory and run `eval.py`.
