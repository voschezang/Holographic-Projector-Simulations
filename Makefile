ADDRESS := markv@stbc-g2.nikhef.nl
PROJECT_DIR := /project/detrd/markv

jupyter:
	jupyter notebook

ssh:
	ssh $(ADDRESS)
	# emacs: use SPC f f /sshx:nikhef

# CUDA
init-path:
	export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}

deps:
	# make sure python, python-pip are installed
	pip --user install virtualenv

vm-update:
	scp -r {Makefile,util.py,plot.py,halton.py} $(ADDRESS):$(DIR)
