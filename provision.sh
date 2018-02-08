# die on error
set -e

sudo apt-get update

sudo apt-get -y install libatlas-base-dev libatlas-dev lib{blas,lapack}-dev gfortran
# install conda, see http://conda.pydata.org/docs/travis.html
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
      wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh
else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda_deps='pip numpy scipy'
conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION $conda_deps
source activate test-environment

# other dependencies
pip install -r requirements.txt

# finally our build
python setup.py build_ext --inplace
