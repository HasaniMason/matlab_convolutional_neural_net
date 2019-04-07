# Matlab Code for Image Classification

This repo contains several Matlab programs which can be used for building convolutional neural networks for image classification.
The included code contains several features:
* Handling imbalanced datasets via weighted Bagging (Bootstrap Aggregation)
* K-fold Cross Validation
* Hyperparameter Optimization
* Finding poorly predicted instances
* Classifying unlabelled data (semi-supervised or active learning)
* Recalibrating CNNs with new data (effectively transfer learning)

The code is organized into six stages. Only the 3rd Stage is required: it contains the basic code needed to build CNNs.
The first two stages are used for hyperaprameter optimization. The 4th through 6th stages are used for validation and handling new data.
The stages are:
1. Optimize the hyperparameters to maximize accuracy over a single partition of the dataset.
2. Further refine the hyperparameters and ensure that the optimizations produce consistent results over all the data partitions via reverse Cross-Validation.
3. Build deployment-ready CNNs via Cross-Validation.
4. Find instances that the deployment-ready CNNs failed to consistently classify via a voting schema.
5. Classify unlabelled data (for semi-supervised or active learning).
6. Recalibrate built models to incorporate new data.


In order for the code to work, each class of images must be in their own folder. Additionally, it is recommended to resize all the images
prior to running the code and to keep the resized images separate from the full sized images. An example structure is shown below.

* /full_size_class_1
* /full_size_class_2
* /working/class_1 (may call working whatever you want, just make it the active path)
* /working/class_2

When recalibrating or classifying unlabeled images, the additional folders are needed.

* /temp/ (contains unlabelled data)
* /temp_resize/ (contains resized unlabelled data)
* /class_1 (will contain output from Stage 5)
* /class_2
* /recalibration/class_1 (for stage 6, may call recalibration whatever, just make it the active path)
* /recalibration/class_2
