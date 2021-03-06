# torchSV
Pytorch implement of [DeepSV](https://github.com/CSuperlei/DeepSV)


## Note: this repo is not completed and has been archived. 

Used `variant graph` instead.

what is `vg`? find it [here](https://github.com/vgteam/vg/wiki)

For SV calling and evaluation, use the [Toil-vg](https://github.com/vgteam/toil-vg/wiki/Genotyping-Structural-Variants) is much better (use `--container None` to run on local nodes ).


### Why I do this ?
I got structural variants from long-read sequencing, and I'd like to infer them using short-read data.

I need a tool that could infer the genotype status given the locations of structural variants.   

However, the source code of the original version of `DeepSV` is not working. 

Here comes the pytorch version of `DeepSV`. 




## Installation and Dependency

- pytorch (> 1.0)
- pillow
- scikit-image
- matplotlib
- numpy
- pandas

## Usage
### Step 1: convert Deletions to Images
need two input: 
1. bam file: postion sorted bam file
2. bed file: convert a vcf file to bed file, only first 3 column are needed.

3. run
```shell
# deletion, need to prepare the bed file 
python preprocess/Generate_Deletion_Images.py --bam test.bam --bed deletion.bed --del_images del
# non  (regions outside deletion), prepare the bed file in advance
python preprocess/Generate_Deletion_Images.py --bam test.bam --bed non_deleltion.bed --del_images non_del
```

### Step 2: train_test_split

Once all the `pileup images` are generated, the rest become easy. 

```python
from dataset import TrainValTestSplit
tvt = TrainValTestSplit("del","non_del")
tvt.to_csv("csv")
```

### Step 3: train the model
```shell
python model/train.py 
```

### Step 4: inference
```shell
python model/eval.py 
```

### Step 5: convert to matrix


## QA
## Contact
