
# Understanding the Geometry and Nature of Specific Interactions in Mixtures 

# Abstract 

The energy of a molecule’s vibrational modes changes with its chemical environment, leading to shifts in characteristic IR bands observed by FTIR spectroscopy. In mixtures, changes in concentration can strengthen or weaken non-covalent interactions, altering the geometry of molecular complexes and shifting IR absorption frequencies. These subtle structural variations can influence important properties such as solubility, reactivity, and binding affinity.

This work aims to understand how the geometry of small-molecule complexes responds to concentration-dependent interactions. It combines quantum mechanical calculations with machine learning models—specifically Graph Neural Networks and Variational Autoencoders—to predict IR frequency shifts based on molecular geometry. Understanding these geometries is crucial for applications in biological systems, catalysis, and formulation science.

# Introduction

Chemical bonds in molecules vibrate at different frequencies depending upon their chemical environment. With increasing concentrations of substances in a mixture the non covalent interactions of a molecule change as the molecule interacts with other species, altering the geometry and the vibrations of the molecule. The most simple example of this can be in a mixture of water, where adding acetone disrupts the hydrogen bonding of water and causes the carbonyl of acetone to vibrate at slightly different frequencies than it would on its own.

The aim of this project is to use FTIR spectroscopy, Quantum Mechanical (QM) calculations and Machine Learning (ML) methods to understand the changing geometry of molecules in mixtures due to changes in non covalent interactions. A simple starting system of water and acetone is used that causes a shift of the carbonyl peak at around 1700cm-1 with increasing water content and hydrogen bond strength. 

In the case of the water acetone system, QM methods are used to sample different geometries of water at distances of between 2-4 angstroms from acetone. This provides features including frequency data, dipole moment, reduced mass, intensity, energy, geometry, coulomb matrices to be used with ML models.  With the inclusion of experimental FTIR data of the acetone mixtures as the target, the geometry of the acetone water complex can be reverse engineered using Graph Neural Networks.

# Method

Mixtures of acetone and water were prepared across a range of volume ratios from 10:90 to 90:10 (acetone:water) and the FTIR spectrum was taken for each. The FTIR has been collected on a Nicolet iD7 with a resolution of 4cm-1. The carbonyl peak of each concentration is determined using Voigt peak fitting and the difference between the experimental peak and the QM calculated vibrations is the target for Machine Learning. 

(voigt_peak_centre.py) 

The QM training data has been created as below, 

1. Optimise the xyz complexe of acetone and water

2. Center the oxygen on the acetone carbonyl to 0,0,0

3. Create ~1000 geometries of water around the acetone. Setting parameters for the hydrogen bond lengths and water angles.
(create_initial_geometries(G).ipynb)

![water angles in training data ](./angle_plot.png)

![distance from acetone oxygen to each hydrogen in training data](./lengths_plot.png)

4. Run QM using Psi4, calculating vibrational frequencies and other features.
(run_QM.py)

5. Filter the geometries that produce useful carbonyl related vibrations and create a graph for use in ML.
(create_graph.py)

6. A conditional Graph VAE was trained to generate new water geometries around the fixed acetone molecule.  Penalties where applied to the O-H bond length and H-O-H angles and Kullback-Leibler divergence was used with warm up scheduling.
(VAE.py)

7. Geometries are predicted using the VAE for each of the experimental FTIR concentration ranges.
(create_geometries.py)

# Background
FTIR blue and red shifts, 

Dilution of acetone with either water or carbon tetrachloride shifts the carbonyl band red or blue, respectively

https://assets.thermofisher.com/TFS-Assets/CAD/Application-Notes/AN50733_E.pdf

"The strong bathochromic shifts observed on methanol OH and acetone CO stretch IR bands are related to hydrogen bonds between these groups. Factor analysis separates the spectra into four acetone and four methanol principal factors." 
https://doi.org/10.1063/1.1790431

"Analysis  of  IR  spectra  of  ethylene  glycol  shows  that  there are only a few contributing bands with solidly fixed vibrational frequencies,  which  only  change  in  relative  intensity  when temperature is changed. It did not show any clear evidence of an intrinsic frequency shift indicating the gradual weakening of hydrogen bonding interaction. Only the relative population of species,  e.g.,  strongly  bonded  and  dissociated  or  much  more weakly  bonded  groups,  seems  to  be  changing.  IR  spectra  of acetone   in   a   mixed   solvent   of   CHCl3/CCl4with   varying composition  also  show  that  intrinsic  IR  frequency  does  notshift  appreciably.  Instead,  only  the  relative  contributions  of highly overlapped adjacent bands are changing."
https://doi.org/10.1366/000370210792434396

# Results
The FTIR plots of the carbonyl peak of the mixtures show a shift of ±5cm-1 with increasing acetone content in Results 1. 

![FTIR of carbonyl peak with increasing acetone, showing acetone carbonyl peak for reference](./carbonyl_peak_of_increasing_acetone_content.png)
Results 1. FTIR of carbonyl peak with increasing acetone, showing acetone carbonyl peak for reference

The training features were produced with QM using acetone and water at different geometries. Results 2 shows how the position of a hydrogen atom in the water molecule varies across the energy landscape of the training dataset.

![Training data shows the position of hydrogen on the energy landscape](./energy_training_data.png)
Results 2. Training data shows the position of hydrogen on the energy landscape

The GraphVAE model was validated on its ability to reconstruct water acetone geometries. The focus was on maintaining realistic O–H bond lengths (~0.96 Å) and H–O–H bond angle (~104.5°) across different concentration conditions.

![training and validation loss](./training_and_validation_loss.png)
Results 3. training and validation loss

![angle error](./angle_error.png)
Results 4. angle error

![bond error](./bond_length_error.png)
Results 5. bond error

The VAE was trained with alternating KL scheduling as seen in VAE2.py leading to the angle and bond error loss seen in Results 4 and 5. The bond error stabilised to around 0.5 angstroms while the angle error was larger, out by up to 15 degrees.

![predict OH bond angle](./predict_OH_bond_angle.png)
Results 6. predicting the H–O–H bond angle

![predict OH bond length](./predict_OH_bond_length.png)
Results 7. predicting the OH bond length

The VAE predicted geometries of up to 100 degrees away from the standard H-O-H bond angle suggesting further work is required to improve the model. Some concentration indexes showed smaller deviance, Results 6.

The prediction of the O-H bond length was better than for the angles, with errors up to 1 angstrom away from the expected bond length.

# Further work 

* The bond angle loss weighting should be increased to improve the accuracy of this property. 

* Look at using alternative constraints in the decoder. 

* Remove poor geometries from the initial QM dataset  

* Include water acetone complexes with varied covalent bond lengths.

* Run QM on predicted geometries and loop this back into the model. 

* Compare other graph neural network architectures

* include more QM training data

* include SAPT energies

* look at other useful complexes, monoethanolamine water for carbon dioxide capture.


# References

https://cs229.stanford.edu/proj2017/final-reports/5244394.pdf

https://www.spectroscopyonline.com/view/five-reasons-why-not-every-peak-shift-in-infrared-ir-spectra-indicates-a-chemical-structure-change

Graph neural networks for materials science and
chemistry
https://www.nature.com/articles/s43246-022-00315-6.pdf

Infrared Spectra Prediction Using Attention-Based Graph Neural Networks. Digital Discovery, Royal Society of Chemistry.
https://doi.org/10.1039/D3DD00254C

Excess Gibbs Free Energy Graph Neural Networks for Predicting Composition-Dependent Activity Coefficients of Binary Mixtures. arXiv preprint.
https://arxiv.org/abs/2407.18372

Representation Learning with a β-Variational Autoencoder for Infrared Spectroscopy. 
https://www.researchgate.net/publication/361453151

Anomaly Detection in Fourier Transform Infrared Spectroscopy of Pharmaceutical Tablets Using Variational Autoencoders. Chemometrics and Intelligent Laboratory Systems, 
https://doi.org/10.1016/j.chemolab.2023.104781

Infrared Spectroscopy of Acetone–Water Liquid Mixtures: Molecular Organization and Hydrogen Bonding. 
https://pubmed.ncbi.nlm.nih.gov/15267555

Spectroscopy from Machine Learning by Accurately Representing Molecular Structures. Nature Communications
https://www.nature.com/articles/s41467-023-36957-4

Learning Molecular Mixture Properties Using Chemistry-Aware Graph Neural Networks.
https://doi.org/10.1103/PRXEnergy.3.023006

Auto-Encoding Variational Bayes.
https://arxiv.org/abs/1312.6114

Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules.
https://doi.org/10.1021/acscentsci.7b00572

A Generative Model for Molecular Distance Geometry.
https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5c4c7bd16c6d0bd1780b9-Abstract.html

GemNet: Universal Directional Graph Neural Networks for Molecules.
https://arxiv.org/abs/2106.08903
 






