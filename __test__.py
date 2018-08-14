from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib

viz = Visdom()
# assert viz.check_connection()
#
# try:
#     import matplotlib.pyplot as plt
#     plt.plot([1, 23, 2, 4])
#     plt.ylabel('some numbers')
#     viz.matplot(plt)
# except BaseException as err:
#     print('Skipped matplotlib example')
#     print('Error message: ', err)

#画出随机的散点图
import time
Y = np.random.rand(100)
old_scatter = viz.scatter(
    X=np.random.rand(100, 2),
    Y=(Y[Y > 0] + 1.5).astype(int),
    opts=dict(
        legend=['Didnt', 'Update'],
        xtickmin=-50,
        xtickmax=50,
        xtickstep=0.5,
        ytickmin=-50,
        ytickmax=50,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    ),
)

time.sleep(5)

#对窗口进行更新,包括标注,坐标,样式等
viz.update_window_opts(
    win=old_scatter,
    opts=dict(
        legend=['Apples', 'Pears'],
        xtickmin=0,
        xtickmax=1,
        xtickstep=0.5,
        ytickmin=0,
        ytickmax=1,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    ),
)
