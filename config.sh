
apt-get install tk-dev python-tk
pip install cffi
pip install pandas
pip install pycocotools
pip install cython
pip install opencv-python
pip install requests
bash lib/build.sh
cd ../
python train.py --dataset csv --csv_train data/annotations.csv --csv_classes data/classes.csv --csv_val data/annotations.csv --epochs 10

