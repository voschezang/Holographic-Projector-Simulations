ADDRESS := markv@stbc-g2.nikhef.nl
PROJECT_DIR := /project/detrd/markv

jupyter:
	jupyter notebook

vm-update:
	scp -r {Makefile,util.py,plot.py,halton.py} $(ADDRESS):$(DIR)

ssh:
	ssh $(ADDRESS)

deps:
	# make sure python, python-pip are installed
	pip --user install virtualenv


# CUDA
init-path:
	export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}

# g2
# ssh-g2:
	# ssh markv@stbc-g2.nikhef.nl
	# sh -c 'cd /project/detrd/markv'
	# dirs ~/src, /data/holoprojector
	# for emacs, use SPC f f /sshx:nikhef
