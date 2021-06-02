#####################################################
# takes predictions from Inception_test.py and true
# mc event positions and puts predictions in the
# right unit (mm), creating a file used to compare
# the different posrec algorithms
#####################################################

import numpy as np

predictions = np.loadtxt("predictions.dat", delimiter = ' ', usecols = [0,1,2])
mc = np.loadtxt("mc_true.dat", delimiter = ' ', usecols = [0,1,2])

print("Number of events in prediction and mc files : ", predictions.shape, mc.shape)

nevt = len(predictions[:,0])
print("nevt =  ",nevt)

pred_x = []
pred_y = []
pred_z = []
mc_x = []
mc_y = []
mc_z = []

for j in range(nevt):
    pred_x.append((predictions[j,0]*2 - 1)*840)
    pred_y.append((predictions[j,1]*2 - 1)*840)
    pred_z.append((predictions[j,2]*2 - 1)*840)

    mc_x.append((mc[j,0]*2 - 1)*840)
    mc_y.append((mc[j,1]*2 - 1)*840)
    mc_z.append((mc[j,2]*2 - 1)*840)

full_pos = np.zeros((nevt,14))
print("ourput file shape : ", full_pos.shape)

for i in range(nevt):
    #predictions of CNN
    full_pos[i,0] = pred_x[i]
    full_pos[i,1] = pred_y[i]
    full_pos[i,2] = pred_z[i]
    #MC true positions
    full_pos[i,3] = mc_x[i]
    full_pos[i,4] = mc_y[i]
    full_pos[i,5] = mc_z[i]
    #Rho
    full_pos[i,6] = (pred_x[i]*pred_x[i] + pred_y[i]*pred_y[i])**(0.5)
    full_pos[i,7] = (mc_x[i]*mc_x[i] + mc_y[i]*mc_y[i])**(0.5)
    #Rho2=Rho/850**2
    full_pos[i,8] = (pred_x[i]*pred_x[i] + pred_y[i]*pred_y[i])/(850*850)
    full_pos[i,9] = (mc_x[i]*mc_x[i] + mc_y[i]*mc_y[i])/(850*850)
    #R
    full_pos[i,10] = (pred_x[i]*pred_x[i] + pred_y[i]*pred_y[i] + pred_z[i]*pred_z[i])**(0.5)
    full_pos[i,11] = (mc_x[i]*mc_x[i] + mc_y[i]*mc_y[i] + mc_z[i]*mc_z[i])**(0.5)
    #R3 = R/850**3
    full_pos[i,12] = ((pred_x[i]*pred_x[i] + pred_y[i]*pred_y[i] + pred_z[i]*pred_z[i])**(1.5))/(850*850*850)
    full_pos[i,13] = ((mc_x[i]*mc_x[i] + mc_y[i]*mc_y[i] + mc_z[i]*mc_z[i])**(1.5))/(850*850*850)

print("full_pos.shape", full_pos.shape)
np.savetxt("full_pred.dat", full_pos, delimiter = ' ')
