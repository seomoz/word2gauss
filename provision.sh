# die on error
set -e

sudo apt-get update

sudo apt-get -y install libatlas-base-dev libatlas-dev lib{blas,lapack}-dev gfortran
sudo pip install conda
conda_deps='pip numpy scipy'
conda create -p $HOME/py --yes $conda_deps "python=$TRAVIS_PYTHON_VERSION"
export PATH=$HOME/py/bin:$PATH
pip install -r requirements.txt

mkdir ~/git
(
    cd ~/git
    git clone https://github.com/seomoz/vocab.git
    (
        cd ~/git/vocab
        pip install -r requirements.txt
        sudo make install
    )
)
python setup.py build_ext --inplace
