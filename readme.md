
# Many tasks make light work: Learning to localise medical anomalies from multiple synthetic tasks


## Environment installation: 
 - Once that's done, create a virtual environment with ```make_virtual_env.sh```
 - Activate the environment with ```source multitask_method_env/bin/activate```
 - Set paths for input/output files in ```multitask_method/paths.py```

Alternatively, you can use the .devcontainer to run the code in a docker container which creates a virtual environment from the ```requirements.txt``` file.

## Data

The HCP dataset is available at https://www.humanconnectome.org/study/hcp-young-adult

The BraTS 2017 dataset is available at https://www.med.upenn.edu/sbia/brats2017/registration.html

The ISLES 2015 dataset is available at https://www.smir.ch/ISLES/Start2015

The VinDr-CXR dataset is available at https://physionet.org/content/vindr-cxr/1.0.0/


## Preprocessing

Preprocessing scripts are available in the ```multitask_method/preprocessing``` folder.
Don't forget to set the paths for preprocessed data and predictions to be saved in ```multitask_method/paths.py```

## Training
With the environment activated, run ```python train.py ABSOLUTE_PATH_TO_EXPERIMENT_CONFIG FOLD_NUMBER```

Experiment configs used for the paper are in the ```experiments``` folder.

## Prediction and Evaluation

To generate predictions on the test set, run ```python predict.py ABSOLUTE_PATH_TO_EXPERIMENT_CONFIG```.

To evaluate the predictions at the normal resolution, run ```python eval.py ABSOLUTE_PATH_TO_EXPERIMENT_CONFIG```.

Result metrics are saved in the predictions folder as ```results.json```.

To evaluate the predictions at CRADL's resolution, run ```python cradl_eval.py ABSOLUTE_PATH_TO_EXPERIMENT_CONFIG```.

## Reproducibility

### Brain

Download the datasets from the above links
Run the brain preprocessing script ```multitask_method/preprocessing/brain_preproc.py```

When experimenting with training on T tasks, run:

For F in range(0, 5CT) run
```python train.py <path_to_repo>/experiments/exp_HCP_low_res_T_train.py F```
To produce predictions run:
```python predict.py <path_to_repo>/experiments/exp_HCP_low_res_T_train.py F```
To evaluate the predictions run:
```python eval.py <path_to_repo>/experiments/exp_HCP_low_res_T_train.py F```
### VinDr-CXR
Download the datasets from the above links
Run the brain preprocessing script ```multitask_method/preprocessing/vindr_cxr_preproc.py```
When experimenting with training on T tasks, run:
For F in range(0, 5CT) run
```python train.py <path_to_repo>/experiments/exp_VINDR_low_res_T_train.py F```
To produce predictions run:
```python predict.py <path_to_repo>/experiments/exp_VINDR_low_res_T_train.py F```
To evaluate the predictions run:
```python eval.py <path_to_repo>/experiments/exp_VINDR_low_res_T_train.py F```
