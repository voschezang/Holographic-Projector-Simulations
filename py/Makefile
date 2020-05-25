PY := python3

jupyter:
	jupyter notebook

init:
	echo "IMG_DIR = '../img'" > _img_dir.py

plot:
	$(PY) main.py

deps:
	# make sure python, python-pip are installed
	pip --user install -r requirements.txt

test:
	$(PY) -m unittest -v test_util.py
