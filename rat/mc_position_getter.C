/////////////////////////////////////////////////////////////
// File taking as input a database file and writes a file
// contaning the event info needed to train CNN
// (x, y, z) coordinates of events
// This output file is to be given to data_crafter.py so
// alongside the PMT file from charge_getter.C
// so that 16x16 charge matrix could be crafted
/////////////////////////////////////////////////////////////

#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TString.h"
#include "TSystem.h"
#include "RAT/DS/Root.hh"
#include "RAT/DS/CAL.hh"
#include "RAT/DS/PMT.hh"
#include "RAT/DEAPStyle.hh"
#include "RAT/DetectorConfig.hh"
#include "RAT/PMTInfoUtil.hh"
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"
#include <fstream>

void mc_position_getter(){

  ofstream fout("output_file_which_will_contain_evt_POS_coordinates.dat",ofstream::out); // format : x y z

  TChain ch("T");
  ch.Add("/home/timcabos/scratch/posreccnn_compilator/ar40/0to200.cal.root");

  cout << "Entries = " << ch.GetEntries() << endl;

  TTree* T = (TTree*) ch.GetTree();
  RAT::DS::Root* ds = new RAT::DS::Root();
  T->SetBranchAddress("ds",&ds);

  //Retrieving MC positions
  for (int entry = 0; entry < T->GetEntries(); entry++) {
    cout << "Entries = " << entry << endl;
    T->GetEntry(entry);
    if (!ds->ExistCAL()) {continue; }
    if (!ds->ExistMC())  {continue; }

    RAT::DS::MC *mc = ds->GetMC();
    RAT::DS::MCParticle *mcparticle = mc->GetMCParticle(0);//apparently, only 1 particle here
    double mcx = mcparticle->GetPosition().X();
    double mcy = mcparticle->GetPosition().Y();
    double mcz = mcparticle->GetPosition().Z();

    fout << entry << " "
     << mcx << " "
     << mcy << " "
     << mcz << " "
     << endl;
  }
}
