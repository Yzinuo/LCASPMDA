# LCASPMDA

**LCASPMDA: a computational model for predicting potential microbe-drug associations based on Learnable Graph Convolutional Attention Networks and Self-Paced Iterative Sampling Ensemble**

### Dataset
  * MDAD: MDAD has 1373 drugs and 173 microbes with 2470 observed drug-microbe pairs
  * aBiofilm: aBiofilm has 1720 drugs and 140 microbes with 2884 observed drug-microbe pairs
  
### Data description
* adj: microbe and drug interactions
* drugs: drug name and corresponding id
* microbes/viruses: microbes/viruses name and corresponding id
* drugfeatures: pre-processed drug structure feature matrix for drugs.
* microbefeatures: pre-processed microbe funtional feature matrix for microbes.
* drugsimilarity: integrated drug similarity matrix.
* microbesimilarity: integrated microbe similarity matrix.

### Run Step 
  Run train.py to train the model and obtain the predicted scores for microbe-drug associations.


### Requirements 
  - python==3.9.5
  - pytorch==1.11.0 
  - tqdm==4.64.0
  - scikit-learn==1.1.1
  - numpy==1.22.3
  - scipy==1.9.3
