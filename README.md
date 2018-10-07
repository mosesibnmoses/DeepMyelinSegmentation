# DeepMyelinSegmentation
Irrespective of initial causes of neurological diseases, these disorders usually exhibit two key pathological changes – axonal loss or demyelination or a mixture of the two. Therefore, vigorous quantification of myelin and axons is essential in studying these diseases. However, the process of quantification has been labor intensive and time-consuming due to the requisite manual segmentation of myelin and axons from microscopic nerve images. As a part of Artificial Intelligence (AI) development, deep learning has been utilized to automate certain human skills, such as image analysis. This study describes the development of a Convolutional Neural Network (CNN) – based approach to segment images of mouse nerve cross sections. We adapted the U-Net architecture and availed ourselves of manually-produced segmentation data accumulated over many years in our lab. These images ranged from normal nerves to those afflicted by severe myelin and axon pathologies, thus maximizing the trained model’s ability to recognize atypical myelin structures. Morphometric data produced by applying the trained model to additional images was then compared to manually obtained morphometrics. The former effectively shortened the time consumption in the morphometric analysis with excellent accuracy in axonal density and g-ratio. However, we were not able to completely eliminate manual refinement of the segmentation product. We also observed small variations in axon diameter and myelin thickness within 9.5%. Nevertheless, we learned alternative ways to improve the accuracy through the study. Overall, greatly increased efficiency in the CNN-based approach out-weighs minor limitations that will be addressed in future studies, thus justifying our confidence in its prospects.



## To train a new model:
put training data in raw/train with files titled x, x_mask.
build the data by running data.py
run train.py with 2 arguments name of model to be saved and number of epochs you want to train it (e.g python train.py newmyelin.h5 100)

## To contunue training a model:
You can resume training an existing model by running extra_training.py with arguments model_in, model_out, num_epochs (e.g oldmyelin.h5 newmyelin.h5 100)

## To apply our model:
simply run find_myelin.py after putting test images in raw/test directory and building test data by running data.py
