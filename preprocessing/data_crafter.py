######################################################
# Creating CSV from root ntuple in order to train
# CNN models used in posreccnn.py
# Normalisation procedure should be improved
######################################################

import numpy as np
from sklearn.preprocessing import RobustScaler

#paths
pmt_path = "path_to_pmt_grid_file"
pmt_coordinates_file = "All_pmts.csv"
ntuple_name = "ntuple_name"
path = "path_to_files_generated_with_charge_getter_and_mc_position_getter"
pos_file = "charge_getter.C_file.dat"
charge_coordinates_file = "mc_position_getter.dat"

#initialisation
pmt_coordinates = np.loadtxt(pmt_path + pmt_coordinates_file, delimiter = ";")
charge_coordinates = np.loadtxt(path + charge_coordinates_file, delimiter = " ", usecols=[0,1,2,3,4])
evt_pos = np.loadtxt(path + pos_file, delimiter = " ", usecols=[0,1,2,3])
n_evt = int(evt_pos[-1,0])
print("number of event = ", int(evt_pos[-1,0]), len(evt_pos), int(charge_coordinates[-1,0]))

#charge normalization ----> see later
#max_Q = np.max(charge_coordinates[:,2])
#print("maxQ :", max_Q)
#charge_coordinates[:,2] = charge_coordinates[:,2]/max_Q
#print("new_maxQ :", np.max(charge_coordinates[:,2]))

#initialising arrays
pmt_array =  np.zeros((n_evt, 4, 256)) # structure: [pmt_number, charge, costh_pmt, phi_pmt]
img_array = np.zeros((n_evt,16,16))
fake_evt = []

#parameters for charge augmentation, if augment, comment out lines 36-40 AND 81-89
#new_min = 0.
#new_max = 1.
#low_limit = 0.001
#high_limit = 0.6

## Retrieving and reshaping data
i=0
for ievt in range(n_evt):
    print("Processing event number "+str(ievt))
    pmt_array[ievt][1,:] = 0
    pmt_array[ievt][2,:] = np.NaN
    pmt_array[ievt][3,:] = np.NaN
    pmt_array[ievt][0,255] = 255
    pmt_array[ievt][2,255] = 0  #?? No Used so it does not really matter
    pmt_array[ievt][3,255] = 0  #?? Could not leave nan => bug

    #retrieve PMT ID and angular position
    tmp_Q = []
    while charge_coordinates[i,0] == ievt:
        if charge_coordinates[i,0] == ievt :
            #ordering charge by PMT ID
            tmp_Q.append(charge_coordinates[i, :])
            i+=1
        else :
            print("tmp_Q constructed for event "+str(ievt))
            continue
    tmp_Q = np.asarray(tmp_Q)
    if len(tmp_Q) :
        ordered_Q = tmp_Q[np.argsort(tmp_Q[:,1])]
    else :
        print("No event number "+str(ievt))
        fake_evt.append(ievt)
        continue

    print("Filling coordinates for event "+str(ievt))
    for ipmt in range(len(pmt_coordinates)):
        pmt_array[ievt][0, ipmt] = pmt_coordinates[ipmt, 0]
        pmt_array[ievt][2, ipmt] = pmt_coordinates[ipmt, 1]
        pmt_array[ievt][3, ipmt] = pmt_coordinates[ipmt, 2]

    print("Filling charge seen by PMTs for event "+str(ievt))
    for npmt in range(len(ordered_Q)):
        pmt_array[ievt][1, int(ordered_Q[npmt,1])] = ordered_Q[npmt, 2]

    # Data augmentation
#    for iqval in range(len(pmt_array[ievt][1])):
#        if pmt_array[ievt][1,iqval] < low_limit:
#            pmt_array[ievt][1,iqval] = new_min
#        elif low_limit <= pmt_array[ievt][1,iqval] < high_limit:
#            pmt_array[ievt][1,iqval] = new_min + (pmt_array[ievt][1,iqval]-low_limit)*(new_max-new_min)/(high_limit-low_limit)
#            pass
#        else:
#            pmt_array[ievt][1,iqval] = new_max

## Splitting into trainning and testing samples, removing fake events -xhere no charge have bees seen- and saving dat files
Q_notround = np.reshape(pmt_array[:,1],(n_evt,16*16)) #as we want a 16x16 matrix for every event
#mc_pos = np.reshape(evt_pos,(n_evt, 3))
mc_pos = np.array(evt_pos[:,1:])
print("number of event = ", int(evt_pos[-1,0]), len(evt_pos), int(charge_coordinates[-1,0]))
print("Number of fake events ", len(fake_evt))


fold_generator = np.random.RandomState(0)
indices_full = range(n_evt)
print("indice's list size :", len(indices_full))
indices = np.setdiff1d(indices_full, fake_evt)
train_indices = fold_generator.choice(range(len(indices)), size=int(len(indices)/2), replace=False)
print("len indices and size train: ", len(indices),int(len(indices)/2), len(train_indices))
test_indices = np.setdiff1d(range(len(indices)),train_indices)
print("len indices and size train: ", len(test_indices))

print("loading train and test samples ...")
train_Q_tmp = Q_notround[train_indices]
test_Q_tmp = Q_notround[test_indices]
train_pos = mc_pos[train_indices]
test_pos = mc_pos[test_indices]
print("train and test samples loaded !")

print("Q before transformation : ", train_Q_tmp)
np.savetxt("untransformed_events_pmt_Q_TRAIN_from_"+ntuple_name+".dat", train_Q_tmp, delimiter= " ")
np.savetxt("untransformed_events_pmt_Q_TEST_from_"+ntuple_name+".dat", test_Q_tmp, delimiter= " ")

# Applying new kind of normalisation to augment contrast, the transformer should
# be generated from the training set and then appliedbto both sets.
transformer = RobustScaler().fit(train_Q_tmp)
train_Q = transformer.transform(train_Q_tmp)
test_Q = transformer.transform(test_Q_tmp)
print("final train Q :", train_Q)

print("mc pos len vs Q :", len(train_pos), np.shape(train_pos), len(train_Q),np.shape(train_Q), len(test_pos),np.shape(test_pos), len(test_Q), np.shape(test_Q))

np.savetxt("events_mc_position_TRAIN_from_"+ntuple_name+".dat", train_pos, delimiter= " ")
np.savetxt("events_pmt_Q_TRAIN_from_"+ntuple_name+".dat", train_Q, delimiter= " ")
np.savetxt("events_mc_position_TEST_from_"+ntuple_name+".dat", test_pos, delimiter= " ")
np.savetxt("events_pmt_Q_TEST_from_"+ntuple_name+".dat", test_Q, delimiter= " ")
np.savetxt("xpos_mc.dat", train_pos[0], delimiter = " ")
np.savetxt("ypos_mc.dat", train_pos[1], delimiter = " ")
np.savetxt("zpos_mc.dat", train_pos[2], delimiter = " ")

np.savetxt("train_indices.dat", train_indices, delimiter = " ")
np.savetxt("test_indices.dat", test_indices, delimiter = " ")
dump(transformer, open('charger_transformer.pkl', 'wb'))            #Store the transformation applied to normalize the charge (SHould always be fitted on the training data !)
