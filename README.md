# PosRecCNN

4 directories containing all codes needed to train CNNs and evaluate their performances

1. rat/ : <br/>
Three files :<br/>
  rat/event_generator_posreccnn.mac : generating MC events from cedar calling the posreccnn.py processor <br/>
  rat/charge_getter.C : creating from a given database file, a file containing all the PMT info needed to then create the 16x16 charge matrices in order to train CNNs <br/>
  rat/mc_position_getter.C : creating from a given database file, a file containing the MC event positions (x, y, z) <br/>

2. preprocessing/ : <br/>
Four files : <br/>
  preprocessing/data_crafter.py : takes files generated from the rat/ files and reshape then into 16x16 matrices that can be given to CNNs <br/>
  preprocessing/charge_plotter.py : plots one 16x16 matrix, can be used to compare different normalisation procedures <br/>
  preprocessing/All_pmts.csv : containing coordinates of the PMTs <br/>
  preprocessing/PMT_spatial_grid.txt : spatial grid of the PMTs <br/>

3. training_models/ : <br/>
Two files : <br/>
  training_models/Inception_trainning.py (from Lucas Doria) : trains CNNs using data.crafter.py files <br/>
  training_models/Inception_testing.py (from Lucas Doria) : tests CNNs using data.crafter.py files <br/>

4.postprocessing/ : <br/>
Two files : <br/>
  postprocessing/desaler.py : used to put CNN predictions in the right units and shape so that posrec_compare.C. can run <br/>
  postprocessing/posrec_compare.C plots a bunch of graphs that can be used to evaluate posrec algorithms perfomances <br/>
