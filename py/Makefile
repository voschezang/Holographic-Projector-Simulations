PY := python3
IMG_DIR := ../img
# SEQUENCE_DIR is relative to IMG_DIR
SEQUENCE_DIR := sequence-
OUTPUT_NAME_Y := $(IMG_DIR)/ani-y.mp4
OUTPUT_NAME_Z := $(IMG_DIR)/ani-z.mp4

.PHONY: jupyter

jupyter:
	jupyter notebook

init:
	# IMG_DIR is a global constant
	echo "IMG_DIR = '$(IMG_DIR)'" > _img_dir.py

plot:
	echo '$(IMG_DIR)/$(SEQUENCE_DIR)y'
	make clean-plot
	$(PY) main.py --sequence_dir $(SEQUENCE_DIR)

animate:
	rm -f $(OUTPUT_NAME_Y) $(OUTPUT_NAME_Z)
	# convert -quality 100 $(IMG_DIR)/$(SEQUENCE_DIR)y/*_amp.png $(OUTPUT_NAME_Y)
	# -r framerate (input/output)
	# -f force format (file extension)
	# -crf quality, e.g. 15-25
	# -v quiet
	ffmpeg -v 0 -framerate 3 -i $(IMG_DIR)/$(SEQUENCE_DIR)y/%06d_.png -vcodec libx264 -crf 25 $(OUTPUT_NAME_Y)
	# ffmpeg -r 10 -s 1920x1080 -i $(IMG_DIR)/$(SEQUENCE_DIR)y/%06d_.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $(IMG_DIR)/y.mp4
	ffmpeg -v 0 -framerate 4 -i $(IMG_DIR)/$(SEQUENCE_DIR)z/%06d_.png -vcodec libx264 -crf 25 $(OUTPUT_NAME_Z)

plot-animate:
	make plot animate

surf:
	# plot surface
	python3 surf.py

experiment:
	python3 experiment.py

clean:
	rm -rf tmp_pkl/*
	rm -rf tmp_local/*
	make clean-plot

clean-plot:
	rm -rf $(IMG_DIR)/*.png
	mkdir -p $(IMG_DIR)/$(SEQUENCE_DIR)y
	mkdir -p $(IMG_DIR)/$(SEQUENCE_DIR)z
	rm -rf $(IMG_DIR)/$(SEQUENCE_DIR)y/*
	rm -rf $(IMG_DIR)/$(SEQUENCE_DIR)z/*

deps:
	# make sure python, python-pip are installed
	pip --user install -r requirements.txt

test:
	$(PY) -m unittest -v test_util.py
