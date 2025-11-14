# MNSG-Net

1. Environment setup
   pip install -r requirements.txt

2. Change datasets paths
   Find the following lines in `train_MNSG-Net.py`:

   ```python
   DATASET_NAME = "OSF"
   TRAIN_DATA_PATH = "../../{}/train".format(DATASET_NAME)
   ```

​       Replace them with your own dataset path.

