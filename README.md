# GMML-project
This is the project for GMML (Geometrical Methods in Machine Learning) course in Skoltech.
Authors: Sergey Makarychev, Alexander Rozhnov

This project is about a new direction of SPD matrix non-linear learning in a deep models.

This project is based on the article "A Riemannian Network for SPD Matrix Learning" ( see https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14633/14371 for detailes) by Zhiwu Huang, Luc Van Gool.

For implemeting we used formulas from the article and also original matlab code from the following author's GitHub: https://github.com/zzhiwu/SPDNet


Everything is implemented in Python. As for additional modules, we used: Numpy, Sklearn, Scipy=1.17.01 and Pymanopt library. 

Usage:

Step 1: Place the AFEW SPD data under the folder "./data/afew/". The AFEW SPD data can be downloaded from http://www.vision.ee.ethz.ch/~zzhiwu/

Step 2: Load spdnet_train_afew.ipynb for a simple example. If you use this code in the first time, just use commented lines for preprocessing matlab data to numpy arrays. This code will create data, that will be in numpy format and already shuffled. 

Step 3: Enjoy! 
