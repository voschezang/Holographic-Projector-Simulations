ADDRESS := markv@stbc-g2.nikhef.nl
PROJECT_DIR := /project/detrd/markv
MNT_DIR := tmp
REMOTE_DIR := nikhef:/project/detrd/markv/Holographic-Projector/tmp

.PHONY: matlab test jupyter

jupyter:
	make -C py jupyter

build:
	# alias
	make -C cuda build

build-run:
	make -C cuda build-run

run:
	make -C cuda run
	# make zip

plot:
	if [ ! -d "$(MNT_DIR)" ]; then echo "Remote dir is not mounted"; exit 1; fi
	make -C py plot

animate:
	make -C py animate

surf:
	make -C py surf

init:
	make -C py init

test:
	make -C cuda test
	make -C py test

remote-run:
	sh remote_run.sh

remote-run-plot:
	make remote-run plot

ssh:
	ssh -Y $(ADDRESS)
	# emacs: use SPC f f /sshx:nikhef

deps:
	make -C py deps

matlab:
	# ./../../matlab/R2016b/bin/matlab -nodisplay -nosplash -nodesktop -r matlab/holostatistics.m
	# cd matlab && ./../../../matlab/R2016b/bin/matlab -nodisplay -nosplash -nodesktop -r holostatistics.m
	# cd matlab && ./../../../matlab/R2016b/bin/matlab -nodisplay -nosplash -nodesktop -r holofunctions.m
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
