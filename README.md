# BloodCellCounter-Fastrcnn_fpn_CustomDataSet
Test to count the number of platelets, red blood cells (RBCs), and white blood cells (WBCs) in blood sample photos, using https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu/dataset/3 as the training and test dataset.

Installation:

Download the project folder to your hard drive and extract it.

Download the file needed to train the model from the Roboflow website: https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu/dataset/3 (COCO JSON option).

Following the instructions on the Roboflow download page, you will obtain a folder named: Blood-Cell-Detection-3 with three subfolders: train, valid, and test.

You need a Roboflow access key, which is... It's free and allows access to other information and documentation.

You need an environment where the related modules are installed, or you can create a new environment by installing the following:

pip install torch

pip install torchvision

pip install requests

pip install packaging

pip install pyparsing

pip install cycler

pip install python-dateutil

pip install kiwisolver

pip install importlib_resources

A requirements.txt file is attached.

Train and obtain the model (the model is larger than 150MB, so I haven't been able to upload it to the project folder on GitHub).

python train.py

Five epochs are sufficient; the number of the record being processed appears on the screen to show the progress.

At the end, the model is saved in the file fasterrcnn_fpn.pth.

Test:

python test.py

Images appear on the screen for comparison. Images with platelets, RBCs, and WBCs, with boxes labeled by Roboflow, and the corresponding boxes for the model's detection on that image.

Images of the first three images are included.

![Figure 1](https://github.com/ablanco1950/BloodCellCounter-Fastrcnn_fpn_CustomDataSet/blob/main/Figure_1.png)

![Figure 1](https://github.com/ablanco1950/BloodCellCounter-Fastrcnn_fpn_CustomDataSet/blob/main/Figure_2.png)

The following summary is obtained:

Platelets labeled = 36 predicted = 38

RBCs labeled = 398 predicted = 397

WBCs labeled = 37 predicted = 37

Total Blood Cells labeled = 471 predicted = 472

If you want to obtain the final balance without checking each image, you can change the SHOW_OPTION parameter to NO in instruction 22 of the test.py program.

OBSERVATIONS:

The model was obtained by trying to fit the boxes labeled by Roboflow. This means that cells located at the edges and not fully represented are not detected. However, when querying the Roboflow test model, the results were correct. https://universe.roboflow.com/team-roboflow/blood-cell-detection-1ekwu, if they are detected. Since I don't know if each image is part of a set of images, these cells that appear cut off at the edges of the image, if counted, would double the count because they would appear in another image, although if they are not recognized they would not be included in the total count.
