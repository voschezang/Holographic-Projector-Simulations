ADDRESS := markv@stbc-g2.nikhef.nl
PROJECT_DIR := /project/detrd/markv
MNT_DIR := tmp
REMOTE_DIR := nikhef:/project/detrd/markv/Holographic-Projector/tmp

jupyter:
	jupyter notebook

build:
	# alias
	make -C cuda build

build-run:
	make build && make run

run:
	make -C cuda run
	make zip

zip:
	zip tmp/out.zip tmp/out.txt

py:
	python3 main.py -r

plot:
	python3 main.py

remote-run:
	sh remote_run.sh

remote-run-plot:
	make remote-run plot

ssh:
	ssh -Y $(ADDRESS)
	# emacs: use SPC f f /sshx:nikhef

# CUDA
init-path:
	export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}

profile:
	make build
	nvprof ./$(EXE)

vprofile:
	make build
	nvvp ./$(EXE)

setup-profiler:
	modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

deps:
	# make sure python, python-pip are installed
	pip --user install -r requirements.txt

vm-update:
	scp -r {Makefile,util.py,plot.py,halton.py} $(ADDRESS):$(DIR)

mount:
	# note the `/` at the end of my_dir/
	sshfs $(REMOTE_DIR)/ $(MNT_DIR)
	# sshfs nikhef:/project/detrd/markv/Holographic-Projector/test/ $(MNT_DIR)

	# in case of Input/Output error
	# find pid
	# -$ pgrep -lf sshfs
	# -$ kill -9 myPid
	# umount -f $(MNT_DIR)

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
