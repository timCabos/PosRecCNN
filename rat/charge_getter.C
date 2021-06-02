/////////////////////////////////////////////////////////////
// File taking as input a database file and writes a file
// contaning the PMT info needed to train CNN
// This output file is to be given to data_crafter.py
// alongside event position file from mc_position_getter.C
// output file so that 16x16 charge matrix could be crafted
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

void charge_getter(){

  ofstream fout("output_file_which_will_contain_PMT_info.dat",ofstream::out); // format : PMT_ID totalQ costh phi

  TChain ch("T");
  ch.Add("database_file.cal.root");


  cout << "Entries = " << ch.GetEntries() << endl;

  TTree* T = (TTree*) ch.GetTree();
  //TTree* T = ch->CopyTree();
  RAT::DS::Root* ds = new RAT::DS::Root();
  T->SetBranchAddress("ds",&ds);

  //PMT Positions
  RAT::PMTInfoUtil pmtInfo;
  RAT::DetectorConfig * detconf = RAT::DetectorConfig::GetDetectorConfig();
  vector<TVector3> PMTpositions = detconf->GetPMTPositions();
  double deg = 180.0/M_PI;
  double costheta,phi;
  cout << "Entries = " << T->GetEntries() << endl;
  for (int entry = 0; entry < T->GetEntries(); entry++) {
    T->GetEntry(entry);
    if (!ds->ExistCAL()) {continue; }
    if (!ds->ExistMC())  {continue; }

    for (int p = 0; p < ds->GetCAL(0)->GetPMTCount(); p++) {
      RAT::DS::PMT* pmt = ds->GetCAL(0)->GetPMT(p);
      // real PMT positions
      costheta = PMTpositions[pmt->GetID()].CosTheta();
      phi      = PMTpositions[pmt->GetID()].Phi();
      //translate phi/costheta to bins
      //pmtPos->SetBinContent((int)(10*phi),(int)(10*costheta),pmt->GetTotalQ());
      pmt->GetTotalQ();

      fout << entry << " "
           << pmt->GetID() << " "
	   << pmt->GetTotalQ() << " "
	   << costheta << " "
	   << phi << " "
	   << endl;

      cout << "Processing entry no :" << entry << endl;
    }
  }
}
