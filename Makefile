.PHONY: test
test: install
	python3 -m better_einsum

.PHONY: setup
setup:
	python3 -m pip install -U setuptools pip

.PHONY: install
install: setup
	python3 -m pip install -Ue .

.PHONY: clean
clean:
	rm -rf ./dist ./build
	-find . -name "*.pyc" -delete
	-C:/GnuWin32/bin/find.exe . -name "*.pyc" -delete
	-find . -name "__pycache__" -delete
	-C:/GnuWin32/bin/find.exe . -name "__pycache__" -delete

.PHONY: upload
upload: clean test
	python3 setup.py sdist bdist_wheel
	python3 -m pip install -U --ignore-installed twine
	twine upload dist/*
