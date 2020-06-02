PY := python3
IMG_DIR := ../img
# SEQUENCE_DIR is relative to IMG_DIR
SEQUENCE_DIR := sequence-
OUTPUT_NAME := $(IMG_DIR)/y.mp4

jupyter:
	jupyter notebook

init:
	# IMG_DIR is a global constant
	echo "IMG_DIR = '$(IMG_DIR)'" > _img_dir.py

plot:
	echo '$(IMG_DIR)/$(SEQUENCE_DIR)y'
	rm -rf $(IMG_DIR)/*.png
	mkdir -p $(IMG_DIR)/$(SEQUENCE_DIR)y
	rm -rf $(IMG_DIR)/$(SEQUENCE_DIR)y/*
	$(PY) main.py --sequence_dir $(SEQUENCE_DIR)

animate:
	rm -f $(OUTPUT_NAME)
	# convert -quality 100 $(IMG_DIR)/$(SEQUENCE_DIR)y/*_amp.png $(OUTPUT_NAME)
	# -r framerate (input/output)
	# -f force format (file extension)
	# -crf quality, e.g. 15-25
	# -v quiet
	ffmpeg -framerate 5 -i $(IMG_DIR)/$(SEQUENCE_DIR)y/%06d_amp.png -vcodec libx264 -crf 25 $(OUTPUT_NAME)
	# ffmpeg -r 10 -s 1920x1080 -i $(IMG_DIR)/$(SEQUENCE_DIR)y/%06d_amp.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $(IMG_DIR)/y.mp4

deps:
	# make sure python, python-pip are installed
	pip --user install -r requirements.txt

test:
	$(PY) -m unittest -v test_util.py
