# die on error
set -e

sudo apt-get update

sudo apt-get -y install libatlas-base-dev libatlas-dev lib{blas,lapack}-dev gfortran
sudo pip install conda
conda_deps='pip numpy scipy'
conda create -p $HOME/py --yes $conda_deps "python=$TRAVIS_PYTHON_VERSION"

pip install -r requirements.txt

mkdir ~/git
(
    cd ~/git
    git clone https://github.com/seomoz/vocab.git
    (
        cd ~/git/vocab
        pip install -r requirements.txt
        python setup.py build_ext --inplace
    )
)
python setup.py build_ext --inplace
export PATH=$HOME/py/lib:$HOME/py/bin:$PATH
