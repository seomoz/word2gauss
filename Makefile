clean:
	# Remove the build
	rm -rf build dist
	# And all of our pyc files
	rm -f word2gauss/*.pyc test/*.pyc word2gauss/*.so word2gauss/embeddings.cpp
	# And lastly, .coverage files
	rm -f .coverage

test: nose

nose:
	rm -rf .coverage
	nosetests --exe --cover-package=word2gauss --with-coverage --cover-branches -v --cover-erase 

unittest:
	python -m unittest discover -s test

# build inplace for unit tests to pass (since they are run from this
# top level directory we need the .so files to be in the src tree
# when they run.
build: clean
	python setup.py build_ext --inplace
	python setup.py build

install: build
	python setup.py install
