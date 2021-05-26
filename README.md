# DeepDeletion
Pytorch implement of DeepSV

## Installation and Dependency

- pytorch (> 1.0)
- pillow
- scikit-image
- matplotlib
- numpy
- pandas

## Usage
### Step 1: convert Deletion region to Images
need two input: 
1. bam file: get pileup images
2. bed file: deletion coordinates

3. run
```shell
# deletion
python preprocess/Generate_Deletion_Images.py --bam test.bam --bed deletion.bed --del_images del
# non  (regions outside deletion)
python preprocess/Generate_Deletion_Images.py --bam test.bam --bed non_deleltion.bed --del_images non_del
```

### Step 2: train_test_split
```shell
python ....
```

### Step 3: train the model
```shell
python model/train.py --indir datasets
```

### Step 4: inference
```shell
python model/eval.py --indir demo
```

### Step 5: convert to matrix


## QA
## Contact
