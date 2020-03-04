ADDRESS := markv@stbc-g2.nikhef.nl
PROJECT_DIR := /project/detrd/markv
MNT_DIR := tmp
REMOTE_DIR := nikhef:/project/detrd/markv/Holographic-Projector/tmp

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
	pip --user install -r requirements.txt

vm-update:
	scp -r {Makefile,util.py,plot.py,halton.py} $(ADDRESS):$(DIR)

mount:
	# note the `/` at the end of my_dir/
	sshfs $(REMOTE_DIR)/ $(MNT_DIR)
	# sshfs nikhef:/project/detrd/markv/Holographic-Projector/test/ $(MNT_DIR)

umount:
	umount $(MNT_DIR)

rsync:
	# rsync -a $(REMOTE_DIR) $(MNT_DIR)
	# :z to compress, :P to show progress bar
	# --delete  enable file removal
	# --backup --backup-dir=$(MNT_DIR)_backup/
	rsync -azP $(REMOTE_DIR) $(MNT_DIR)
	# rsync -azP nikhef:/project/detrd/markv/Holographic-Projector/test mnt
	# rsync -avP --numeric-ids --exclude='/dev' --exclude='/proc' --exclude='/sys' / root@xxx.xxx.xxx.xxx:/


info:
	lscpu
	lspci -vnn | grep VGA -A 12
	lshw -numeric -C display

add-to-path:
	echo 'export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}'

build:
	# make add-to-path
	# nvcc -o main main.cu
	# -ftz=true // flush dernormal to zero
	#  -ftz=true -prec-div=false -prec-sqrt=false
	/usr/local/cuda-10.1/bin/nvcc -o main main.cu

build-run:
	make build && make run

run:
	./main

py:
	python3 main.py -r

plot:
	python3 main.py -scatter

remote-run:
	sh remote_run.sh

remote-run-plot:
	make remote-run plot
