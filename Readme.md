# Experiment with AST: Asymmetric Student-Teacher Networks for Industrial Anomaly Detection

This is the code to the WACV 2023 paper "[Asymmetric Student-Teacher Networks for Industrial Anomaly Detection](https://arxiv.org/pdf/2210.07829.pdf)" by Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn and Bastian Wandt.

## Getting Started

You will need [Python 3.7.7](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) and installing the packages there.

Install packages with:

```
$ pip install -r requirements.txt
```

## Configure

All configurations concerning data, model, training, visualization etc. can be made in _config.py_. The default configuration will run a training with paper-given parameters for MVTec 3D-AD.

The following steps guide you from your dataset to your evaluation:

* Set your `dataset_dir` in `config.py`. This is the directory which contains the subdirectories of the classes you want to process. It should be configured whether the datasets contains 3D scans (set `use_3D_dataset` in `config.py`). If 3D data is present (`use_3D_dataset=True`), the data structure from [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) is assumed. It can be chosen which data domain should be used for detection (set mode in `config.py`). If the dataset does not contain 3D data (`use_3D_dataset=False`), the data structure from the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) is assumed. Both dataset structures are described in the documentation of `load_img_datasets` in `utils.py`.
* `dilate_mask` in `config.py`: True for 3D setting.
* `localize` in `config.py`: True for anomaly map visualization.
* `preprocessing.py`: It is recommended to pre-extract the features beforehand to save training time. This script also preprocesses 3D scans. Alternatively, the raw images are used in training when setting `pre_extracted=False` in `config.py`.

## Run

* `python train_teacher.py`: Trains the teacher and saves the model to `models/...` . The teacher is trained with data augmentation. The augmented normal training images are saved in `datasets/mvtec_ad/bottle/train/good_aug/...` . Meanwhile the original and augmented images are loaded for training the teacher. There are three data augmentation methods applied to MVTec AD dataset and five for MVTec 3D dataset.
* `python train_student.py`: Trains the student and saves the model to `models/...` . The student is trained without data augmentaion. You may reach out to the the parameter `img_aug` in the file to change the setting. The same goes for the teacher training.
* `python eval.py`: Evaluates the student-teacher-network (image-level results and localization/segmentation).  Additionally, it can create ROC curves, anomaly score histograms, and localization images. The anomaly maps are saved in `viz/maps/...` . The image-level results of max F1 scores and corresponding classification thresholds are displayed. An csv file named `evaluation_results.csv` is saved in the current directory with the predictions of individual images in the test set.

## Comment

* The code support the test images that are merely divided into two classes: normal and anomalous. The images could vary in sizes and aspect ratios. 
* In our experiments, training the teacher with data augmentation and the student without data augmentation results in the best*  performance on the bottle class of MVTec AD dataset. 


## Credits

Some code of an old version of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.

## Citation
Please cite our paper in your publications if it helps your research. Even if it does not, you are welcome to cite the authors.

    @inproceedings { RudWeh2023,
    author = {Marco Rudolph and Tom Wehrbein and Bodo Rosenhahn and Bastian Wandt},
    title = {Asymmetric Student-Teacher Networks for Industrial Anomaly Detection},
    booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
    year = {2023},
    month = jan
    }

## License

This project is licensed under the MIT License.
