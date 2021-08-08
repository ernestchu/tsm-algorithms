# TSM Algorithms
This repo implements several time-scale modification algorithms on audio signals.

## Prepare environment
### Create virtual environment
It's always good to give your python project a dedicated environment (where pip packages installed) via **venv**.
```sh
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```
You'll get something like this in your terminal
```sh
(.venv) username@devicename $
```
If you want to exit the environment
```sh
deactivate
```

You can also check the status by
```sh
which pip
```
This should give you the path to your dedicated pip executable within `.venv`.

### Add IPython kernel
To use this environment we just made in Jupyter notebook, we need to add the kernel to the globally installed Jupyter.
```sh
source .venv/bin/activate # make sure you're using the right Python executable
python -m ipykernel install --user --name tsm
```
Now you're all set. Now you can use the packages installed in this environment in Jupyter notebook. You can also double check by
```sh
jupyter kernelspec list
```

### Uninstall the environment
Remove `.venv` and remove the IPython kernel
```sh
rm -rf .venv
jupyter kernelspec remove tsm
```



