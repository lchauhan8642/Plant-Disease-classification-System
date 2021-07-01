Steps to download and configure Software for the Project: - 

1) Download Anaconda3-5.2.0 
Link: https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe

2)After Installation Completion. Open Anaconda Prompt and run below commands

conda update setuptools
conda update wrapt
pip install tensorflow==2
pip install opencv-contrib-python
pip install msgpack
pip install keras


3)Download Tensorflow-gpu and setup according to available Nvidia graphic card (To run the program on the GPU of the system)
Run Command on Anaconda Prompt: pip install tensorflow-gpu==2.0

Download cuda and cudnn from the below link and setup it.

Setup Link for cudnn: https://www.tensorflow.org/install/gpu 
                                  https://developer.nvidia.com/cudnn
                                 https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
                                 https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

to check everything is installed correctly. Open Anaconda Prompt and run " import tensorflow as tf " and if it shows no error the system is configurated to run the project. 

4) Dataset used "New Plant Disease Dataset"'
Link: https://www.kaggle.com/vipoooool/new-plant-diseases-dataset

Change the main folder name to "dataset". It will include train, valid and test folder.

5) Setup Visual Studio Code(Editor) for python.
(In Visual Studio Code select python3.6.5 as python interperter to run the python files as downloaded above)
Download Extension: - Anaconda Extension Pack
                                   Qt for Python
                                   Python 
                                   Python for VSCode

6) Run main.py file