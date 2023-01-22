# ecg_vit  
1D vision transformer (vit) for ecg interpretable classification  
Code for my [blog post](https://yonigottesman.github.io/ecg/vit/deep-learning/2023/01/20/ecg-vit.html)

## Installation
python version `3.10.5`  
dependencies in `requirements.txt`

## Training
`python train.py /<path>/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/`

## Plot attension
~~~bash
python plot_attention.py  <wfdb_file> <model_weights_path> <lead_index>`  
~~~

For example:  
~~~bash
python plot_attention.py a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/06/066/JS05865 vit_best/ 6
~~~
