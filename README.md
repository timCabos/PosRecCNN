# PosRecCNN

4 directories containing all codes needed to train CNNs and evaluate their performances

1. rat/ :
Three files :
  rat/event_generator_posreccnn.mac : generating MC events from cedar calling the posreccnn.py processor
  rat/charge_getter.C : creating from a given database file, a file containing all the PMT info needed to then create the 16x16 charge matrices in order to train CNNs
  rat/mc_position_getter.C : creating from a given database file, a file containing the MC event positions (x, y, z)

2. preprocessing/ :
Four files : 
  preprocessing/data_crafter.py : takes files generated from the rat/ files and reshape then into 16x16 matrices that can be given to CNNs
  preprocessing/charge_plotter.py : plots one 16x16 matrix, can be used to compare different normalisation procedures
  preprocessing/All_pmts.csv : containing coordinates of the PMTs
  preprocessing/PMT_spatial_grid.txt : spatial grid of the PMTs

3. training_models/ :
Two files :
  training_models/Inception_trainning.py (from Lucas Doria) : trains CNNs using data.crafter.py files
  training_models/Inception_testing.py (from Lucas Doria) : tests CNNs using data.crafter.py files

4. postprocessing/ :
Two files :
  postprocessing/desaler.py : used to put CNN predictions in the right units and shape so that posrec_compare.C. can run
  postprocessing/posrec_compare.C plots a bunch of graphs that can be used to evaluate posrec algorithms perfomances
