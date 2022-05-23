import ROOT
from ROOT import TLorentzVector, TH1F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from bcml4pheno import bcml_model
from bcml4pheno import get_elijah_ttbarzp_cs
from bcml4pheno import get47Dfeatures
from bcml4pheno import get_settings
import scipy.interpolate
import matplotlib.pyplot as plt
from datetime import datetime
from bcml4pheno import cross_section_helper
from xgboost import XGBClassifier


#indxs = [0,25,26,37,16,3,27,28,17,23,39,35,2,18,1,21,34]
indxs = [0, 25, 36, 26, 1, 27, 35, 37, 16, 28, 2, 39, 5, 18, 29, 33, 11, 40, 22, 3]
indxs = indxs[:19]


def PT(TLV):
    return TLV.Pt()

def cross_cleaning(jets, bjets, leptons):
    particles = jets + bjets + leptons
    overlapping = False
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            if (particles[i].DeltaR(particles[j]) < 0.3):
                overlapping = True
                break
    return overlapping


entries = ["eff_ttbarh", "eff_ttbarttbar" ,"eff_ttbarbbar_noh", "eff_Zprime_bbar_250", "eff_Zprime_bbar_350", "eff_zprime_bbar_1000"]
#entries = ["ttbarh", "ttbarttbar" ,"ttbarbbar_noh", "Zprime_bbar_350", "Zprime_bbar_1000", "Zprime_bbar_3000"]
type_entries = ["bkg", "bkg", "bkg", "signal", "signal", "signal" ]
#jobs = [99, 99, 99, 19, 19, 19]
jobs = [10, 10, 10, 10, 10, 8]

cuts = np.zeros((len(entries), 3))

arrs = []

print("Begin of Data Reading")

for n_signal, signal in enumerate(entries):

  #Counters of the number of events that pass each cut. Used for calculating the efficiencies of each cut.
  cut0 = 0
  cut1 = 0
  cut2 = 0

  #Counter used for saving in txt the information of the Gen-particles of the first 10 events. Only used when gentxt == True
  txt_counter = 0

  arr1 = []
  f = ROOT.TFile(signal + ".root", "recreate")

  for ind in range(1,jobs[n_signal]+1):
    directory= str("../from_disco3/with_delphes/" + signal + "/" + signal + "_" + str(ind) + "/Events/run_01/tag_1_delphes_events.root")
    File = ROOT.TChain("Delphes;1")
    File.Add(directory)
    Number = File.GetEntries()

    print("Input: " + signal + "_" + str(ind))

    for i in range(Number):
    	#print("Evento: " + str(i))
        Entry = File.GetEntry(i)

        jets = []
        bjets = []
        electrons = []
        muons = []
        leptons = []
        METs = []

        EntryFromBranch_j = File.Jet.GetEntries()
        for j in range(EntryFromBranch_j):

                BTag = File.GetLeaf("Jet.BTag").GetValue(j)
		        #print("PT = {}, BTag = {}".format(PT,BTag) )
                if (BTag != 1):
                  jet = TLorentzVector()
                  jet_PT, jet_Eta, jet_Phi, jet_M  = File.GetLeaf("Jet.PT").GetValue(j), File.GetLeaf("Jet.Eta").GetValue(j), File.GetLeaf("Jet.Phi").GetValue(j), File.GetLeaf("Jet.Mass").GetValue(j)
                  jet.SetPtEtaPhiM(jet_PT, jet_Eta, jet_Phi, jet_M)
                  jets.append(jet)

                elif (BTag == 1):
                  bjet = TLorentzVector()
                  bjet_PT, bjet_Eta, bjet_Phi, bjet_M  = File.GetLeaf("Jet.PT").GetValue(j), File.GetLeaf("Jet.Eta").GetValue(j), File.GetLeaf("Jet.Phi").GetValue(j), File.GetLeaf("Jet.Mass").GetValue(j)
                  bjet.SetPtEtaPhiM(bjet_PT, bjet_Eta, bjet_Phi, bjet_M)
                  bjets.append(bjet)

        EntryFromBranch_e = File.Electron.GetEntries()
        for j in range(EntryFromBranch_e):
          electron = TLorentzVector()
          electron_PT, electron_Eta, electron_Phi, electron_M  = File.GetLeaf("Electron.PT").GetValue(j), File.GetLeaf("Electron.Eta").GetValue(j), File.GetLeaf("Electron.Phi").GetValue(j), 0.000510998928
          electron.SetPtEtaPhiM(electron_PT, electron_Eta, electron_Phi, electron_M)
          electrons.append(electron)

        EntryFromBranch_mu = File.Muon.GetEntries()
        for j in range(EntryFromBranch_mu):
          muon = TLorentzVector()
          muon_PT, muon_Eta, muon_Phi, muon_M  = File.GetLeaf("Muon.PT").GetValue(j), File.GetLeaf("Muon.Eta").GetValue(j), File.GetLeaf("Muon.Phi").GetValue(j), 0.1056583745
          muon.SetPtEtaPhiM(muon_PT, muon_Eta, muon_Phi, muon_M)
          muons.append(muon)

        EntryFromBranch_MET = File.MissingET.GetEntries()
        for j in range(EntryFromBranch_MET):
          MET = TLorentzVector()
          MET_PT, MET_Eta, MET_Phi, MET_M  = File.GetLeaf("MissingET.MET").GetValue(j), File.GetLeaf("MissingET.Eta").GetValue(j), File.GetLeaf("MissingET.Phi").GetValue(j), 0.0
          MET.SetPtEtaPhiM(MET_PT, MET_Eta, MET_Phi, MET_M)
          METs.append(MET)


        leptons = electrons + muons


        jets = [jet for jet in jets if jet.Pt() > 30 and abs(jet.Eta()) < 5.0]
        bjets = [b for b in bjets if b.Pt() > 30 and abs(b.Eta()) < 2.4]
        leptons = [l for l in leptons if l.Pt() > 20 and abs(l.Eta()) < 2.3]
        cut0 += 1
        if (cross_cleaning(jets, bjets, leptons) == False):
            cut1 += 1
            if (len(bjets) == 4 and len(jets) == 2 and len(leptons) == 1):
                cut2 += 1

                jets.sort(reverse = True, key=PT)
                bjets.sort(reverse = True, key=PT)
                j1, j2 = jets[0], jets[1]
                b1, b2 = bjets[0], bjets[1]


                leptons_tot = np.sum(np.array(leptons))
                MET = np.sum(np.array(METs))
                #indxs = [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18,25,35,36,38,39,40,41,42,43,44,45,46]
            
            

                row = np.array([bjets[0].Pt(), bjets[1].Pt(), bjets[2].Pt(), bjets[3].Pt(), (bjets[0].Eta() - bjets[1].Eta()), (bjets[0].Eta() - bjets[2].Eta()), (bjets[0].Eta() - bjets[3].Eta()), (bjets[1].Eta() - bjets[2].Eta()), (bjets[1].Eta() - bjets[3].Eta()), (bjets[2].Eta() - bjets[3].Eta()), bjets[0].DeltaPhi(bjets[1]), bjets[0].DeltaPhi(bjets[2]), bjets[0].DeltaPhi(bjets[3]), bjets[1].DeltaPhi(bjets[2]), bjets[1].DeltaPhi(bjets[3]), bjets[2].DeltaPhi(bjets[3]), bjets[0].DeltaR(bjets[1]), bjets[0].DeltaR(bjets[2]), bjets[0].DeltaR(bjets[3]), bjets[1].DeltaR(bjets[2]), bjets[1].DeltaR(bjets[3]), bjets[2].DeltaR(bjets[3]), MET.Pt(), np.sum(np.array(leptons)).Pt(), (leptons_tot + MET).Mt(), (bjets[0] + bjets[1]).M(), (bjets[0] + bjets[2]).M(), (bjets[0] + bjets[3]).M(), (bjets[1] + bjets[2]).M(), (bjets[1] + bjets[3]).M(), (bjets[2] + bjets[3]).M(), (bjets[0] + leptons_tot + MET).Mt(), (bjets[1] + leptons_tot + MET).Mt(), (bjets[2] + leptons_tot + MET).Mt(), (bjets[3] + leptons_tot + MET).Mt(), (jets[0] + jets[1]).M(), jets[0].Pt(), jets[1].Pt(), jets[0].DeltaR(jets[1]), bjets[0].DeltaR(leptons_tot), bjets[1].DeltaR(leptons_tot), bjets[2].DeltaR(leptons_tot), bjets[3].DeltaR(leptons_tot), bjets[0].DeltaPhi(leptons_tot), bjets[1].DeltaPhi(leptons_tot), bjets[2].DeltaPhi(leptons_tot), bjets[3].DeltaPhi(leptons_tot), bjets[0].Phi(), bjets[1].Phi(), bjets[0].Eta(), bjets[1].Eta(), jets[0].Phi(), jets[1].Phi(), jets[0].Eta(), jets[1].Eta(), leptons_tot.Phi(), leptons_tot.Eta()])
                #row = np.array([bjets[0].Pt(), bjets[1].Pt(), bjets[2].Pt(), bjets[3].Pt(), jets[0].Pt(), jets[1].Pt(), (bjets[0] + bjets[3]).M(), (bjets[1] + bjets[3]).M(), (bjets[2] + bjets[3]).M(), bjets[0].DeltaR(bjets[3]), bjets[1].DeltaR(bjets[3]), bjets[2].DeltaR(bjets[3]), bjets[3].DeltaR(leptons_tot), MET.Pt()])

                #row = np.array([bjets[0].Pt(), bjets[1].Pt(), bjets[2].Pt(), (bjets[0].Eta() - bjets[1].Eta()), (bjets[0].Eta() - bjets[2].Eta()), (bjets[1].Eta() - bjets[2].Eta()), bjets[0].DeltaPhi(bjets[1]), bjets[0].DeltaPhi(bjets[2]), bjets[1].DeltaPhi(bjets[2]), bjets[0].DeltaR(bjets[1]), bjets[0].DeltaR(bjets[2]), bjets[1].DeltaR(bjets[2]), MET.Pt(), np.sum(np.array(leptons)).Pt(), (leptons_tot + MET).Mt(), (bjets[0] + bjets[1]).M(), (bjets[0] + bjets[2]).M(), (bjets[1] + bjets[2]).M(), (bjets[0] + leptons_tot + MET).Mt(), (bjets[1] + leptons_tot + MET).Mt(), (bjets[2] + leptons_tot + MET).Mt(), (jets[0] + jets[1]).M(), jets[0].Pt(), jets[1].Pt(), jets[0].DeltaR(jets[1]), bjets[0].DeltaR(leptons_tot), bjets[1].DeltaR(leptons_tot), bjets[2].DeltaR(leptons_tot), bjets[0].DeltaPhi(leptons_tot), bjets[1].DeltaPhi(leptons_tot), bjets[2].DeltaPhi(leptons_tot)])
                #row = np.array([bjets[0].Pt(), bjets[1].Pt(), bjets[2].Pt(), bjets[3].Pt(), jets[0].Pt(), jets[1].Pt(), (bjets[0] + bjets[3]).M(), (bjets[1] + bjets[3]).M(), (bjets[2] + bjets[3]).M(), bjets[0].DeltaR(bjets[3]), bjets[1].DeltaR(bjets[3]), bjets[2].DeltaR(bjets[3]), bjets[3].DeltaR(leptons_tot), MET.Pt()])
                #arr1.append(row[indxs])
                arr1.append(row)
            
   

  cut = [cut0, cut1, cut2]
  cuts[n_signal, :] = cut 
        
  arrs.append(arr1)            
  
  

print("End of data reading")

print("Begin of Data Preparation")
arrs = np.array(arrs)
bkg1 = np.array(arrs[0])
bkg2 = np.array(arrs[1])
bkg3 = np.array(arrs[2])
signal1, signal1_vsize = np.array(arrs[3]), np.shape(np.array(arrs[3]))[0]
signal2, signal2_vsize = np.array(arrs[4]), np.shape(np.array(arrs[4]))[0]
signal3, signal3_vsize = np.array(arrs[5]), np.shape(np.array(arrs[5]))[0]

np.savetxt("signal250.txt", signal1)
np.savetxt("signal350.txt", signal2)
np.savetxt("signal1000.txt", signal3)
np.savetxt("bkg1.txt", bkg1)
np.savetxt("bkg2.txt", bkg2)
np.savetxt("bkg3.txt", bkg3)

np.savetxt("number_of_ev.txt", cuts)



print("nevents sgn1: ", np.shape(signal1))
print("nevents sgn2: ", np.shape(signal2))
print("nevents sgn3: ", np.shape(signal3))
print("nevents bkg1: ", np.shape(bkg1))
print("nevents bkg2: ", np.shape(bkg2))
print("nevents bkg3: ", np.shape(bkg3))

print("cuts: ", cuts)





    

