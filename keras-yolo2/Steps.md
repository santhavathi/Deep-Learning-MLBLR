# Step by Step Instructions

## **Object Detection - YOLOv**

### Part I

```
cd Downloads/GitRepo
git clone https://github.com/pjreddie/darknet
cd darknet
make
brew install wget
wget https://pjreddie.com/media/files/yolo.weights
./darknet detect cfg/yolov2.cfg yolo.weights data/k.jpg
```
---

### Part II

#### Install Conda

https://www.anaconda.com/download/#macos
Download Anaconda3-5.0.1-MacOSX-x86_64.pkg, double click on pkg file to install under home directory /Users/santhavathis

Installation under /Users/santhavathis/anaconda3

```.bash_profile
export PATH="/Users/santhavathis/anaconda3/bin:$PATH"

python --version
Python 3.6.3 :: Anaconda, Inc.

conda --version
conda 4.3.30
```

#### Install Virtual Environment

```
cd /Users/santhavathis/anaconda3 
```
https://conda.io/docs/user-guide/getting-started.html
Create virtualenv dl and install opencv and keras packages in it
```
conda create --name dl opencv keras

(dl) Santhavathis-MacBook-Air:dl santhavathis$ source activate dl
(dl) Santhavathis-MacBook-Air:dl santhavathis$ conda --info envs
(dl) Santhavathis-MacBook-Air:dl santhavathis$ pip2 freeze | grep opencv
```

#### Enable Virtual Environment for Jupyter

http://anbasile.github.io/programming/2017/06/25/jupyter-venv/
https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/

```
(dl) Santhavathis-MacBook-Air:dl santhavathis$ pip install ipykernel
(dl) Santhavathis-MacBook-Air:dl santhavathis$ ipython kernel install --user --name=dl
cd /Users/santhavathis/Library/Jupyter/kernels/dl
vi kernel.json => Ensure the path points to the virtualenv
{
 "argv": [
  "/Users/santhavathis/anaconda3/envs/dl/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "dl",
 "language": "python"
}
```

Then invoke ipython notebook and change the kernel for the .ipynb file.
ipython notebook  or jupyter notebook => Kernel => Choose the kernel "dl"


#### Download the train and test images and annotations

http://images.cocodataset.org/zips/train2014.zip
http://images.cocodataset.org/zips/val2014.zip
http://images.cocodataset.org/annotations/annotations_trainval2014.zip

Unzip train and test images under Downloads/images folder
```
ls -l /Users/santhavathis/Downloads/images/val2014/ | head -4
total 13124168
-rw-rw-r--@ 1 santhavathis  staff  213308 Aug 16  2014 COCO_val2014_000000000042.jpg
-rw-rw-r--@ 1 santhavathis  staff  383651 Aug 16  2014 COCO_val2014_000000000073.jpg
-rw-rw-r--@ 1 santhavathis  staff  176151 Aug 16  2014 COCO_val2014_000000000074.jpg

ls -l /Users/santhavathis/Downloads/images/train2014/ | head -4

Unzip annotations
ls -l /Users/santhavathis/Downloads/annotations
total 1650056
-rw-rw-r--@ 1 santhavathis  staff   66782097 Sep  1  2017 captions_train2014.json
-rw-rw-r--@ 1 santhavathis  staff   32421077 Sep  1  2017 captions_val2014.json
-rw-rw-r--@ 1 santhavathis  staff  332556225 Sep  1  2017 instances_train2014.json
-rw-rw-r--@ 1 santhavathis  staff  160682675 Sep  1  2017 instances_val2014.json
-rw-r--r--@ 1 santhavathis  staff  170733465 Sep  1  2017 person_keypoints_train2014.json
-rw-r--r--@ 1 santhavathis  staff   81637509 Sep  1  2017 person_keypoints_val2014.json
```

#### Convert the annotations from COCO to VOC format

https://gist.github.com/chicham/6ed3842d0d2014987186#file-coco2pascal-py
Under /Users/santhavathis/anaconda3/envs/dl
vi coco2pascal.py and copy the contents from the above link, change below lines
```
from path import path
as
from path import Path as path

print out_name
as
print(out_name)

print instance[‘file_name’]
as
print(instance[‘file_name’])

(dl) Santhavathis-MacBook-Air:dl santhavathis$ pip install baker cytoolz lxml
https://github.com/jaraco/path.py
(dl) Santhavathis-MacBook-Air:dl santhavathis$ pip install path.py

(dl) Santhavathis-MacBook-Air:dl santhavathis$ mkdir /Users/santhavathis/Downloads/val
(dl) Santhavathis-MacBook-Air:dl santhavathis$ mkdir /Users/santhavathis/Downloads/train

(dl) Santhavathis-MacBook-Air:dl santhavathis$ python3 coco2pascal.py create_annotations /Users/santhavathis/Downloads val /Users/santhavathis/Downloads/val

(dl) Santhavathis-MacBook-Air:~ santhavathis$ ls -l /Users/santhavathis/Downloads/val/ | wc -l
```
One xml file is created for each image, the format of which is as below

```
<annotation>
	<folder>VOC2014</folder>
	<filename>COCO_val2014_000000012343.jpg</filename>
	<source>
		<database>MS COCO 2014</database>
		<annotation>MS COCO 2014</annotation>
		<image>Flickr</image>
	</source>
	<size>
		<width>427</width>
		<height>640</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>motorcycle</name>
		<bndbox>
			<xmin>192.44</xmin>
			<ymin>90.11</ymin>
			<xmax>576.35</xmax>
			<ymax>423.28000000000003</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<bndbox>
			<xmin>311.79</xmin>
			<ymin>1.95</ymin>
			<xmax>543.57</xmax>
			<ymax>335.11</ymax>
		</bndbox>
	</object>
</annotation>

(dl) Santhavathis-MacBook-Air:dl santhavathis$ ls -l /Users/santhavathis/Downloads/val | wc -l
   40073
(dl) Santhavathis-MacBook-Air:dl santhavathis$ ls -l /Users/santhavathis/Downloads/images/val2014/ | wc -l
   40505

(dl) Santhavathis-MacBook-Air:dl santhavathis$ python3 coco2pascal.py create_annotations /Users/santhavathis/Downloads train /Users/santhavathis/Downloads/train
```

#### Install YOLOv2 model

Install pre-requisites
https://github.com/aleju/imgaug
conda list => and install all missing prerequisites mentioned in above link
```
pip install tqdm
pip install scikit-image
pip freeze | grep scikit-image

(dl) Santhavathis-MacBook-Air:dl santhavathis$ pip install git+https://github.com/aleju/imgaug
pip freeze | grep imgaug

(dl) Santhavathis-MacBook-Air:dl santhavathis$ conda list | grep keras
keras                     2.1.5                    py36_0

https://github.com/keras-team/keras/issues/9349 => Issue is encountered when running the keras-yolo2 notebook with eras version 2.1.5, hence installed version 2.1.2

(dl) Santhavathis-MacBook-Air:dl santhavathis$ conda install keras==2.1.2

(dl) Santhavathis-MacBook-Air:dl santhavathis$ git clone https://github.com/experiencor/keras-yolo2.git

Copy pre-trained weights
cp /Users/santhavathis/Downloads/GitRepo/darknet/yolo.weights /Users/santhavathis/anaconda3/envs/dl/keras-yolo2/.

mkdir /Users/santhavathis/logs

(dl) Santhavathis-MacBook-Air:dl santhavathis$ cd keras-yolo2/
And invoke “jupyter notebook” and change Kernel to “dl” for notebook Yolo Step-by-Step.ipynb

Change the paths in Yolo Step-by-Step.ipynb as follows
wt_path = 'yolo.weights'                      
train_image_folder = ‘/Users/santhavathis/Downloads/images/train2014/'
train_annot_folder = ‘/Users/santhavathis/Downloads/train/'
valid_image_folder = ‘/Users/santhavathis/Downloads/images/val2014/'
valid_annot_folder = ‘/Users/santhavathis/Downloads/val/‘
```

#### Run Object Detection

Now run the .ipynb file. The model.fit takes 50+hrs for 1 epoch on laptop, with the last 4 layers shuffled and using only validation data of 40k images. 

Hence did not run the “Randomize weights of the last layer” section and used the yolo.weights.

Run the “Perform detection on image” directly with the above weights and was able to detect image and video.