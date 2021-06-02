//==============================================================================
// Plotting macro for DEAP position reconstruction performance studies
// Author: Simon Viel (Carleton)
// No Cuts, compare posreccnn predictions with MBLikelihood and TimeFit2 ones
// using the prediction file from Inception_test.py (in trainning_models/)
// reshaped so that positions are in mm (using descaler.py)!
//==============================================================================
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TString.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TH1D.h"
#include "TGraphAsymmErrors.h"
#include "TH2D.h"
#include "TLegend.h"
//#include "TLine"

#include <iostream>
#include <vector>

void posrec_compare() {
  TString plotdir = "/Users/tim/Desktop/DEAP_3600/Codes/plot_macro/PosRecCNN/position_resolution_tests/test_with_luca_data/";
  TString posrec_path = plotdir;
  TString topdir = "/Users/tim/Desktop/DEAP_3600/Codes/Inception/model_from_luca/data_from_sviel_ntuple/ntuples/ar39_ntp/2019/";
  TString ntuple_name = "ar39_nominal_00000.ntp.root";

  const bool fullCuts = true;
  if(fullCuts) plotdir += "_with_cuts_broken";
  plotdir += "/";
  gSystem->mkdir( plotdir , kTRUE );

  //Creating Canvas and color code
  TCanvas* c1 = new TCanvas("c1", "Canvas", 10, 10, 800, 600);

  gStyle->SetOptStat(0);

  Double_t stops[10] = { 0.0000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000};
  Double_t red[10]   = { 1.0000, 0.9764, 0.9956, 0.8186, 0.5301, 0.1802, 0.0232, 0.0780, 0.0592, 0.2082};
  Double_t green[10] = { 1.0000, 0.9832, 0.7862, 0.7328, 0.7492, 0.7178, 0.6419, 0.5041, 0.3599, 0.1664};
  Double_t blue[10]  = { 1.0000, 0.0539, 0.1968, 0.3499, 0.4662, 0.6425, 0.7914, 0.8385, 0.8684, 0.5293};
  TColor::CreateGradientColorTable(10, stops, red, green, blue, 255);

  TLine* redline = new TLine(-850, -850, 551, 551);
  redline->SetLineColor(6);
  redline->SetLineWidth(2);

  //TLine* absredline = new TLine(0, 0, 850, 850);
  TLine* absredline = new TLine(0, 0, 1, 1);
  absredline->SetLineColor(6);
  absredline->SetLineWidth(2);

  TString cutstext = "{qPE > 0}";
  if(fullCuts) cutstext = "{Full analysis cuts}";

  TString maincuts = "qPE > 0";
  TString cuts = "1";

  // CREATING GRAPHS
  //Compare
  TH1D* hT = new TH1D("hT", "", 200, -2000, 2000);
  TH1D* hM = new TH1D("hM", "", 200, -2000, 2000);
  TH1D* hP = new TH1D("hP", "", 200, -2000, 2000);
  //Z
  TH2D* hT_Z = new TH2D();
  TH2D* hM_Z = new TH2D();
  TH2D* hP_Z = new TH2D();
  //Rho
  TH2D* hT_Rho = new TH2D();
  TH2D* hM_Rho = new TH2D();
  TH2D* hP_Rho = new TH2D();
  //Rho2
  TH2D* hT_Rho2 = new TH2D();
  TH2D* hM_Rho2 = new TH2D();
  TH2D* hP_Rho2 = new TH2D();
  //R
  TH2D* hT_R = new TH2D();
  TH2D* hM_R = new TH2D();
  TH2D* hP_R = new TH2D();
  //R3
  TH2D* hT_R3 = new TH2D();
  TH2D* hM_R3 = new TH2D();
  TH2D* hP_R3 = new TH2D();

  // CNN
  TTree *pred_tree = new TTree("posrec_pos", "prediction positions");
  pred_tree->ReadFile(posrec_path+"full_pred.dat", "xpos:ypos:zpos:mcxpos:mcypox:mczpos:Rho:mcRho:Rho2:mcRho2:R:mcR:R3:mcR3", ' ');
  //Z
  hP_Z->GetXaxis()->SetLimits(-850,600);
  hP_Z->GetYaxis()->SetLimits(-850,600);
  hP_Z->SetTitle("posreccnn z position vs mc z position");
  hP_Z->GetXaxis()->SetTitle("Z mc generated [mm]");
  hP_Z->GetYaxis()->SetTitle("PosRecCNN Z [mm]");
  hP_Z->GetYaxis()->SetTitleOffset(1.2);
  hP_Z->Draw();
  c1->Update();
  pred_tree->Draw("zpos:mczpos", "", "COLZSAME0");
  redline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "zpos_resolution_posreccnn.png");

  pred_tree->Draw("zpos - mczpos >> hP","","same");
  hP->Scale(1./hP->Integral());
  hP->SetLineColor(7);
  hP->SetLineWidth(2);

  //Rho
  hP_Rho->GetXaxis()->SetLimits(-850,600);
  hP_Rho->GetYaxis()->SetLimits(-850,600);
  hP_Rho->SetTitle("posreccnn Rho position vs mc Rho position");
  hP_Rho->GetXaxis()->SetTitle("Rho mc generated [mm]");
  hP_Rho->GetYaxis()->SetTitle("PosRecCNN Rho [mm]");
  hP_Rho->GetYaxis()->SetTitleOffset(1.2);
  hP_Rho->Draw();
  c1->Update();
  pred_tree->Draw("Rho:mcRho", "", "COLZSAME0");
  redline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rho_resolution_posreccnn.png");

  pred_tree->Draw("Rho - mcRho >> hP","","same");
  hP->Scale(1./hP->Integral());
  hP->SetLineColor(7);
  hP->SetLineWidth(2);

  //Rho
  hP_Rho2->GetXaxis()->SetLimits(-850,600);
  hP_Rho2->GetYaxis()->SetLimits(-850,600);
  hP_Rho2->SetTitle("posreccnn Rho2 position vs mc Rho2 position");
  hP_Rho2->GetXaxis()->SetTitle("Rho2 mc generated [mm]");
  hP_Rho2->GetYaxis()->SetTitle("PosRecCNN Rho2 [mm]");
  hP_Rho2->GetYaxis()->SetTitleOffset(1.2);
  hP_Rho2->Draw();
  c1->Update();
  pred_tree->Draw("Rho2:mcRho2", "", "COLZSAME0");
  redline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rho2_resolution_posreccnn.png");

  pred_tree->Draw("Rho2 - mcRho2 >> hP","","same");
  hP->Scale(1./hP->Integral());
  hP->SetLineColor(7);
  hP->SetLineWidth(2);

  //R
  hP_R->GetXaxis()->SetLimits(-850,600);
  hP_R->GetYaxis()->SetLimits(-850,600);
  hP_R->SetTitle("posreccnn R position vs mc R position");
  hP_R->GetXaxis()->SetTitle("R mc generated [mm]");
  hP_R->GetYaxis()->SetTitle("PosRecCNN R [mm]");
  hP_R->GetYaxis()->SetTitleOffset(1.2);
  hP_R->Draw();
  c1->Update();
  pred_tree->Draw("R:mcR", "", "COLZSAME0");
  redline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "R_resolution_posreccnn.png");

  pred_tree->Draw("R - mcR >> hP","","same");
  hP->Scale(1./hP->Integral());
  hP->SetLineColor(7);
  hP->SetLineWidth(2);

  //R3
  hP_R3->GetXaxis()->SetLimits(-850,600);
  hP_R3->GetYaxis()->SetLimits(-850,600);
  hP_R3->SetTitle("posreccnn R3 position vs mc R3 position");
  hP_R3->GetXaxis()->SetTitle("R3 mc generated [mm]");
  hP_R3->GetYaxis()->SetTitle("PosRecCNN R3 [mm]");
  hP_R3->GetYaxis()->SetTitleOffset(1.2);
  hP_R3->Draw();
  c1->Update();
  pred_tree->Draw("R3:mcR3", "", "COLZSAME0");
  redline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "R3_resolution_posreccnn.png");

  pred_tree->Draw("R3 - mcR3 >> hP","","same");
  hP->Scale(1./hP->Integral());
  hP->SetLineColor(7);
  hP->SetLineWidth(2);

  // Other graphs
  TFile *f = TFile::Open(topdir+ntuple_name);
  TTree* background = (TTree*) f->Get("data_mc");
  //Z
  hT_Z->GetXaxis()->SetLimits(-850,600);
  hT_Z->GetYaxis()->SetLimits(-850,600);
  hT_Z->SetTitle("timefit2 z position vs mc z position");
  hT_Z->GetXaxis()->SetTitle("Z mc generated [mm]");
  hT_Z->GetYaxis()->SetTitle("TimeFit2 Z [mm]");
  hT_Z->GetYaxis()->SetTitleOffset(1.2);
  hT_Z->Draw();
  c1->Update();
  background->Draw("timefit2Z:mc_z >> hT_Z", maincuts + cuts, "COLZSAME0");
  redline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "zpos_resolution_timefit2.png");

  hM_Z->GetXaxis()->SetLimits(-850,600);
  hM_Z->GetYaxis()->SetLimits(-850,600);
  hM_Z->SetTitle("mblikelihood z position vs mc z position");
  hM_Z->GetXaxis()->SetTitle("Z mc generated [mm]");
  hM_Z->GetYaxis()->SetTitle("MBLikelihood Z [mm]");
  hM_Z->GetYaxis()->SetTitleOffset(1.2);
  hM_Z->Draw();
  c1->Update();
  background->Draw("mblikelihoodZ:mc_z >> hM_Z", maincuts + cuts,"COLZSAME0");
  redline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "zpos_resolution_mblikelihood.png");

  background->Draw("timefit2Z - mc_z >> hT", maincuts + cuts, "same");
  hT->Scale(1./hT->Integral());
  hT->SetLineColor(kGreen+3);
  hT->SetLineWidth(2);

  background->Draw("mblikelihoodZ - mc_z >> hM", maincuts + cuts, "same");
  hM->Scale(1./hM->Integral());
  hM->SetLineColor(2);
  hM->SetLineWidth(2);

  TLegend* legend = new TLegend(0.15,0.7,0.38,0.88);
  legend->SetTextSize(0.04);
  legend->SetLineColor(0);
  legend->SetFillColor(0);
  legend->SetFillStyle(0);

  legend->AddEntry(hM, "MB Likelihood", "L");
  legend->AddEntry(hT, "timefit2", "L");
  legend->AddEntry(hP, "Posrec CNN", "L");

  c1->SetLogy(1);
  hM->SetTitle("recoZ - mc_z  ");
  hM->SetXTitle("recoZ - mc_z [mm]");
  hM->SetYTitle("Entries");
  hM->Draw();
  hT->Draw("same");
  hM->Draw("same");
  hP->Draw("same");
  legend->Draw("same");
  c1->Print(plotdir + "zpos_resolution_log.png");
  c1->SetLogy(0);
  c1->Print(plotdir + "zpos_resolution.png");

  //Rho
  TString timefit2Rho = "sqrt(timefit2X*timefit2X+timefit2Y*timefit2Y)";
  TString mblikelihoodRho = "sqrt(mblikelihoodX*mblikelihoodX+mblikelihoodY*mblikelihoodY)";
  TString timefit2Rho2 = "(timefit2X*timefit2X+timefit2Y*timefit2Y)/(850*850)";
  TString mblikelihoodRho2 = "(mblikelihoodX*mblikelihoodX+mblikelihoodY*mblikelihoodY)/(850*850)";
  TString genRho = "sqrt(mc_x*mc_x+mc_y*mc_y)";
  TString genRho2 = "(mc_x*mc_x+mc_y*mc_y)/(850*850)";

  hT_Rho->GetXaxis()->SetLimits(0,1);
  hT_Rho->GetYaxis()->SetLimits(0,1);
  hT_Rho->SetTitle("timefit2 Rho position vs mc Rho position");
  hT_Rho->GetXaxis()->SetTitle("Rho mc generated [mm]");
  hT_Rho->GetYaxis()->SetTitle("TimeFit2 Rho [mm]");
  hT_Rho->GetYaxis()->SetTitleOffset(1.2);
  hT_Rho->Draw();
  c1->Update();
  background->Draw(timefit2Rho2+":"+genRho2+" >> hT_Rho", maincuts + cuts, "COLZSAME0");
  absredline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rhopos_resolution_timefit2.png");

  hM_Rho->GetXaxis()->SetLimits(0,1);
  hM_Rho->GetYaxis()->SetLimits(0,1);
  hM_Rho->SetTitle("mblikelihood Rho position vs mc Rho position");
  hM_Rho->GetXaxis()->SetTitle("Rho mc generated [mm]");
  hM_Rho->GetYaxis()->SetTitle("MBLikelihood Rho [mm]");
  hM_Rho->GetYaxis()->SetTitleOffset(1.2);
  hM_Rho->Draw();
  c1->Update();
  background->Draw(mblikelihoodRho2+":"+genRho2+" >> hM_Rho", maincuts + cuts, "COLZSAME0");
  absredline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rhopos_resolution_mblikelihood.png");

  background->Draw(timefit2Rho+" - "+genRho+" >> hT", maincuts + cuts, "same");
  hT->Scale(1./hT->Integral());
  hT->SetLineColor(kGreen+3);
  hT->SetLineWidth(2);

  background->Draw(mblikelihoodRho+" - "+genRho+" >> hM", maincuts + cuts, "same");
  hM->Scale(1./hT->Integral());
  hM->SetLineColor(2);
  hM->SetLineWidth(2);


  c1->SetLogy(1);
  hM->GetXaxis()->SetRangeUser(-850, 850);
  hM->SetTitle("recoRho - genRho  " + cutstext);
  hM->SetXTitle("recoRho - genRho [mm]");
  hM->SetYTitle("Entries");
  hM->Draw();
  hT->Draw("same");
  hM->Draw("same");
  hP->Draw("same");
  legend->Draw("same");
  c1->Print(plotdir + "Rhopos_resolution_log.png");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rhopos_resolution.png");


  //R
  TString timefit2R = "sqrt(timefit2X*timefit2X+timefit2Y*timefit2Y+timefit2Z*timefit2Z)";
  TString mblikelihoodR = "sqrt(mblikelihoodX*mblikelihoodX+mblikelihoodY*mblikelihoodY+mblikelihoodZ*mblikelihoodZ)";
  TString timefit2R3 = "sqrt(timefit2X*timefit2X+timefit2Y*timefit2Y+timefit2Z*timefit2Z)**3/(850*850*850)";
  TString mblikelihoodR3 = "sqrt(mblikelihoodX*mblikelihoodX+mblikelihoodY*mblikelihoodY+mblikelihoodZ*mblikelihoodZ)**3/(850*850*850)";
  TString genR = "sqrt(mc_x*mc_x+mc_y*mc_y+mc_z*mc_z)";
  TString genR3 = "sqrt(mc_x*mc_x+mc_y*mc_y+mc_z*mc_z)**3/(850*850*850)";

  hT_R->GetXaxis()->SetLimits(0,1);
  hT_R->GetYaxis()->SetLimits(0,1);
  hT_R->SetTitle("timefit2 R position vs mc R position");
  hT_R->GetXaxis()->SetTitle("R mc generated [mm]");
  hT_R->GetYaxis()->SetTitle("TimeFit2 R [mm]");
  hT_R->GetYaxis()->SetTitleOffset(1.2);
  hT_R->Draw();
  c1->Update();
  background->Draw(timefit2R3+":"+genR3+" >> hT_R", maincuts + cuts, "COLZSAME0");
  absredline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rpos_resolution_timefit2.png");

  hM_R->GetXaxis()->SetLimits(0,1);
  hM_R->GetYaxis()->SetLimits(0,1);
  hM_R->SetTitle("mblikelihood R position vs mc R position");
  hM_R->GetXaxis()->SetTitle("R mc generated [mm]");
  hM_R->GetYaxis()->SetTitle("MBLikelihood R [mm]");
  hM_R->GetYaxis()->SetTitleOffset(1.2);
  hM_R->Draw();
  c1->Update();
  background->Draw(mblikelihoodR3+":"+genR3+" >> hM_R", maincuts + cuts, "COLZSAME0");
  absredline->Draw("same");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rpos_resolution_mblikelihood.png");

  background->Draw(timefit2R+" - "+genR+" >> hT", maincuts + cuts, "same");
  hT->Scale(1./hT->Integral());
  hT->SetLineColor(kGreen+3);
  hT->SetLineWidth(2);

  background->Draw(mblikelihoodR+" - "+genR+" >> hM", maincuts + cuts, "same");
  hM->Scale(1./hT->Integral());
  hM->SetLineColor(2);
  hM->SetLineWidth(2);

  c1->SetLogy(1);
  hM->GetXaxis()->SetRangeUser(0, 1);
  hM->SetTitle("recoR - genR  " + cutstext);
  hM->SetXTitle("recoR - genR [mm]");
  hM->SetYTitle("Entries");
  hM->Draw();
  hT->Draw("same");
  hM->Draw("same");
  hP->Draw("same");
  legend->Draw("same");
  c1->Print(plotdir + "Rpos_resolution_log.png");
  c1->SetLogy(0);
  c1->Print(plotdir + "Rpos_resolution.png");


}
