virtualenv -p python3 multitask_method_env

source multitask_method_env/bin/activate

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install -e .
