apt-get install tk-dev python-tk
pip install cffi
pip install pandas
pip install pycocotools
pip install cython
pip install opencv-python
pip install requests
bash lib/build.sh
cd ..


python visualize.py --dataset csv --csv_classes data/cla.csv  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
python3 visualize.py --dataset csv  --csv_classes data/classes.csv --csv_val data/annotations.csv --model model_final.pt 