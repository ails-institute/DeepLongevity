# DeepLogevity
Deep Learning for ageing analysis in Caenorhabditis elegans micrographs.

Here we compare several deep learning approaches in predicing lifespan and total movement of C. elegans worms. Using this approach we are looking for label-free micrographs cues allowing us to locate age-related pathologies in the model organism.

## Utils folder contains

scanWormFolders.py - script that scans the folder with raw images, calculate the lifespan for each worm and creates a table with lifepan length for all worms in the experiment

wormTrainer.py - script that we used to train InceptionV3 model to classify worms into 2 classes: "long" and "short" lifespan. Data augmentation was implemented and all the parameters are set in the script. It saves the model and the obtained weights.

wormTrainer_DenseNet.py, wormTrainer_MobilNet.py, wormTrainer_NASNet.py - scripts where we used DenseNet, MobilNet, NASNet accordingly to classify "long" and "short" lifespan.

wormTrainer_WormNet.py - script where we implemented a CNN called WormNet.py, which designed by Artur Yakimovich specifically to work with C. elegans images.

wormTester.py - script to evaluate a pretrained model performance with loaded weights on the test dataset.

wormActivator.py - script to visualise the global average pooling layer to  produce a so called class activation map and localize class-specific image regions in unsupervised manner

## WormNet folder contains

Most up-to-date WormNet hyperparameters, as well as requirements.

## HydraNet-colab

Contains Google Colab notebooks containing HydraNet experiments.
