```python
import os
import gudhi
import json
import numpy as np
from functools import partial
import classifier
import sklearn.metrics as metrics
from importlib import reload
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import region
import shapely as sh
import matplotlib.pyplot as plt
from importlib import reload

```


```python
def get_persistence_dict(file_name):
    with open(file_name,'r') as f:
        data_dict=json.load(f)
    return data_dict
```


```python
basic_work_path=os.path.join(os.getcwd(),'result','complex')
complex_list=os.listdir(basic_work_path)
file_list=[]
for complex_ in ['alpha']:
    work_path=os.path.join(basic_work_path,complex_)
    tmp=os.listdir(work_path)
    for file in tmp:
        file_list.append(os.path.join(work_path,file))
```


```python
for file in [file_list[0]]:
    work_path=os.path.join(os.getcwd(),'result','pd')
    data_dict=get_persistence_dict(file)
    for key in data_dict.keys():
        for i, data in enumerate(data_dict[key]):
            gudhi.plot_persistence_diagram(data)
            _,name=os.path.split(file)
            name,_=os.path.splitext(name)
            save_path=os.path.join(work_path,name)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            save_name=os.path.join(save_path,key+'_'+str(i)+'.png')
            plt.savefig(save_name)
```

    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical left == right == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical bottom == top == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical left == right == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical bottom == top == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical left == right == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical bottom == top == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical left == right == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:348: UserWarning: Attempting to set identical bottom == top == 0.0 results in singular transformations; automatically expanding.
      axes.axis([axis_start, axis_end, axis_start, infinity + delta/2])
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    D:\PROJECT\seminconductor\code\lib\site-packages\gudhi\persistence_graphical_tools.py:288: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
      fig, axes = plt.subplots(1, 1)
    


![png](get_each_pd_multi_files/get_each_pd_multi_3_1.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_2.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_3.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_4.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_5.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_6.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_7.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_8.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_9.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_10.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_11.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_12.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_13.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_14.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_15.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_16.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_17.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_18.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_19.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_20.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_21.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_22.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_23.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_24.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_25.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_26.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_27.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_28.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_29.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_30.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_31.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_32.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_33.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_34.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_35.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_36.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_37.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_38.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_39.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_40.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_41.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_42.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_43.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_44.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_45.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_46.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_47.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_48.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_49.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_50.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_51.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_52.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_53.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_54.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_55.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_56.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_57.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_58.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_59.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_60.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_61.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_62.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_63.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_64.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_65.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_66.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_67.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_68.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_69.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_70.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_71.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_72.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_73.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_74.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_75.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_76.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_77.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_78.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_79.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_80.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_81.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_82.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_83.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_84.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_85.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_86.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_87.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_88.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_89.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_90.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_91.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_92.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_93.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_94.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_95.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_96.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_97.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_98.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_99.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_100.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_101.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_102.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_103.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_104.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_105.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_106.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_107.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_108.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_109.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_110.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_111.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_112.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_113.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_114.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_115.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_116.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_117.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_118.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_119.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_120.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_121.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_122.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_123.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_124.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_125.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_126.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_127.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_128.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_129.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_130.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_131.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_132.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_133.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_134.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_135.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_136.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_137.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_138.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_139.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_140.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_141.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_142.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_143.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_144.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_145.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_146.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_147.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_148.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_149.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_150.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_151.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_152.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_153.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_154.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_155.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_156.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_157.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_158.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_159.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_160.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_161.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_162.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_163.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_164.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_165.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_166.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_167.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_168.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_169.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_170.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_171.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_172.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_173.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_174.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_175.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_176.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_177.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_178.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_179.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_180.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_181.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_182.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_183.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_184.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_185.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_186.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_187.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_188.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_189.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_190.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_191.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_192.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_193.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_194.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_195.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_196.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_197.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_198.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_199.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_200.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_201.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_202.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_203.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_204.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_205.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_206.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_207.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_208.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_209.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_210.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_211.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_212.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_213.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_214.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_215.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_216.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_217.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_218.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_219.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_220.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_221.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_222.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_223.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_224.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_225.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_226.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_227.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_228.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_229.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_230.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_231.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_232.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_233.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_234.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_235.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_236.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_237.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_238.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_239.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_240.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_241.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_242.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_243.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_244.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_245.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_246.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_247.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_248.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_249.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_250.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_251.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_252.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_253.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_254.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_255.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_256.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_257.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_258.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_259.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_260.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_261.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_262.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_263.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_264.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_265.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_266.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_267.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_268.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_269.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_270.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_271.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_272.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_273.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_274.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_275.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_276.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_277.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_278.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_279.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_280.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_281.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_282.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_283.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_284.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_285.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_286.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_287.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_288.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_289.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_290.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_291.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_292.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_293.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_294.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_295.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_296.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_297.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_298.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_299.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_300.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_301.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_302.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_303.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_304.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_305.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_306.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_307.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_308.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_309.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_310.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_311.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_312.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_313.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_314.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_315.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_316.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_317.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_318.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_319.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_320.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_321.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_322.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_323.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_324.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_325.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_326.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_327.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_328.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_329.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_330.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_331.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_332.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_333.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_334.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_335.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_336.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_337.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_338.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_339.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_340.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_341.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_342.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_343.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_344.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_345.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_346.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_347.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_348.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_349.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_350.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_351.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_352.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_353.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_354.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_355.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_356.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_357.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_358.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_359.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_360.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_361.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_362.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_363.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_364.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_365.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_366.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_367.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_368.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_369.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_370.png)



![png](get_each_pd_multi_files/get_each_pd_multi_3_371.png)



```python
os.getcwd()
```




    'D:\\PROJECT\\seminconductor\\code\\data2'




```python

```
