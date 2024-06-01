# SAUCE

This is an open source implementation of "SAUCE: A Self-adaptive Update Method For Learned Cardinality Estimators".

### Environments

```shell
# create anaconda environments
conda create --name sauce python=3.9
conda activate sauce

cd SAUCE/
# PyTorch is also contained in requirements.txt. You can also comment it and install it by yourself
pip install -r requirements.txt

# Install dependencies of FACE
cd ./FACE/torchquadMy
pip install .
```

### End-to-end experiments for each datasets

```shell
# You can run an end-to-end experiment by running this conmmand. Some config examples are stored in ./end2end/configs/. You can choose the config file from this python script
python end2end/multi_experiments.py
```
