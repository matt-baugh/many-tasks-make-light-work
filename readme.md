
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

### Inter-dataset blending datasets

The MOOD dataset (for 3D inter-dataset blending) is available at https://www.synapse.org/#!Synapse:syn21343101/wiki/599515

The ImageNET dataset (for 2D inter-dataset blending) is available at https://www.kaggle.com/c/imagenet-object-localization-challenge/overview

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


## Cite this work:

```
@InProceedings{baugh2023manytasks,
    author="Baugh, Matthew
    and Tan, Jeremy
    and M{\"u}ller, Johanna P.
    and Dombrowski, Mischa
    and Batten, James
    and Kainz, Bernhard",
    title="Many Tasks Make Light Work: Learning to Localise Medical Anomalies from Multiple Synthetic Tasks",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
    editor="Greenspan, Hayit
    and Madabhushi, Anant
    and Mousavi, Parvin
    and Salcudean, Septimiu
    and Duncan, James
    and Syeda-Mahmood, Tanveer
    and Taylor, Russell",
    year="2023",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="162--172",
    isbn="978-3-031-43907-0"
}

```

## Acknowledgements

(Some) HPC resources were provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR projects b143dc and b180dc. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683.
