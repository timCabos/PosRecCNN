######################################################
# Plot a 16x16 matrix from a given event feeded to
# CNN models used in posreccnn.py
# Three options : No normalization and two different
# normalization procedures
######################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

charges = np.loadtxt('events_pmt_Q_TRAIN_file_from_data_crafter.py.dat', delimiter = ' ')
#charges = np.loadtxt('untransformed_events_pmt_Q_filefrom_data_crafter.py.dat', delimiter = ' ')
#max = np.max(charges)
#charges_to_plot = charges[int(evt_number)]/max
charges_to_plot = charges[int(evt_number)]
charge_matrix = np.reshape(charges_to_plot, (16,16))

print("16x16",charges_to_plot.shape)


plt.figure(figsize=(15,15))
sns.set_theme()
sns.heatmap(charge_matrix, annot=True, linewidths= 1,)
plt.show()
