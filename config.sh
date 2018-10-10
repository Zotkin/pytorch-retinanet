
apt-get install tk-dev python-tk
pip3 install cffi
pip3 install pandas
pip3 install pycocotools
pip3 install cython
pip3 install opencv-python
pip3 install requests
bash lib/build.sh
cd ../
python train.py --dataset csv --csv_train data/annotations.csv --csv_classes data/classes.csv --csv_val data/annotations.csv --epochs 10

