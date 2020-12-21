ADDRESS := markv@stbc-g2.nikhef.nl
PROJECT_DIR := /project/detrd/markv
MNT_DIR := tmp
REMOTE_DIR := nikhef:/project/detrd/markv/Holographic-Projector/tmp

.PHONY: matlab test

jupyter:
	make -C py jupyter

build:
	make -C cuda build

build-run:
	make -C cuda build-run

run:
	make -C cuda run

plot:
	make -C py plot

animate:
	make -C py animate

surf:
	make -C py surf

init:
	make -C py init

test:
	make -C py test

remote-run:
	sh remote_run.sh

remote-run-plot:
	if [ ! -d "$(MNT_DIR)" ]; then echo "Remote dir is not mounted"; exit 1; fi
	make remote-run plot

ssh:
	ssh -Y $(ADDRESS)
	# emacs: use SPC f f /sshx:nikhef

deps:
	make -C py deps

matlab:
	cd matlab && ./../../../matlab/R2016b/bin/matlab -nodisplay -nosplash -nodesktop

matlab-gui:
	./../../matlab/R2016b/bin/matlab

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

info:
	lscpu
	lspci -vnn | grep VGA -A 12
	lshw -numeric -C display
