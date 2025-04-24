
# Understanding the Geometry and Nature of Specific Interactions in Mixtures 

# Abstract 

Non covalent interactions in mixtures vary with concentration and cause shifts in the vibrational modes of molecular functional groups, seen by IR spectroscopy. It can be suggested that with changing concentration and non covalent interactions, the geometry of a molecule complex changes in a similar way to well known reaction mechanisms such as nucleophilic and electrophilic substitution. The frequency shifts for simple acetone water mixtures vary up to 5cm-1 requiring high predictive accuracy and supporting the idea that the geometries follow a small change in trajectory. Variational auto encoder models can be used with Graph Neural networks that represent the molecular structure of complexes and can make predictions using QM features with the target being experimental IR peaks. Initial results show that the GNN model can predict within this range (residuals predictions). As the input QM geometries keep the covalent bond lengths and angles constant the GNN model includes loss functions for both these properties. Further work is required to include variations in the covalent bond lengths and angles to provide more valid vibrational frequencies.

# Introduction

Chemical bonds in molecules vibrate at different frequencies depending upon their chemical environment. With increasing concentrations of substances in a mixture the non covalent interactions of a molecule change as the molecule interacts with other species, altering the geometry and the vibrations of the molecule. The most simple example of this can be in a mixture of water, where adding acetone disrupts the hydrogen bonding of water and causes the carbonyl of acetone to vibrate at slightly different frequencies than it would on its own.

The aim of this project is to use FTIR spectroscopy, Quantum Mechanical (QM) calculations and Machine Learning (ML) methods to understand the changing geometry of molecules in mixtures due to changes in non covalent interactions. A simple starting system of water and acetone is used that causes a shift of the carbonyl peak at around 1700cm-1 with increasing water content and hydrogen bond strength. 

In the case of the water acetone system, QM methods are used to sample different geometries of water at distances of between 2-4 angstroms from acetone. This provides features including frequency data, dipole moment, reduced mass, intensity, energy, geometry, coulomb matrices to be used with ML models.  With the inclusion of experimental FTIR data of the acetone mixtures as the target, the geometry of the acetone water complex can be reverse engineered using Graph Neural Networks.

# Method
The FTIR has been collected on a Nicolet iD7 with a resolution of 4cm-1. 
acetone = 10.csv with increasing water -> 0.csv = water. 

The peak centres of the mixtures is determined using Voigt peak fitting. 
(voigt_peak_centre.py) 

1. Optimise the xyz complexes of acetone and water

2. Center the oxygen on the acetone carbonyl to 0,0,0

3. Create ~1000 geometries of water around the acetone.
(create_initial_geometries(G).ipynb)

4. Run QM using Psi4, calculating vibrational frequencies and other features.
(run_QM.py)

5. Filter the geometries that produce useful carbonyl related vibrations and create a graph for use in ML.
(create_graph.py)

6. A conditional Graph VAE was trained to generate new water geometries around the fixed acetone molecule.  Penalties where applied to the O-H bond length and H-O-H angles and Kullback-Leibler divergence was used with warm up scheduling.
(VAE.py)



# Background
FTIR blue and red shifts, 

Dilution of acetone with either water or carbon tetrachloride shifts the carbonyl band red or blue, respectively

https://assets.thermofisher.com/TFS-Assets/CAD/Application-Notes/AN50733_E.pdf

"The strong bathochromic shifts observed on methanol OH and acetone CO stretch IR bands are related to hydrogen bonds between these groups. Factor analysis separates the spectra into four acetone and four methanol principal factors." 
https://doi.org/10.1063/1.1790431

"Analysis  of  IR  spectra  of  ethylene  glycol  shows  that  there are only a few contributing bands with solidly fixed vibrational frequencies,  which  only  change  in  relative  intensity  when temperature is changed. It did not show any clear evidence of an intrinsic frequency shift indicating the gradual weakening of hydrogen bonding interaction. Only the relative population of species,  e.g.,  strongly  bonded  and  dissociated  or  much  more weakly  bonded  groups,  seems  to  be  changing.  IR  spectra  of acetone   in   a   mixed   solvent   of   CHCl3/CCl4with   varying composition  also  show  that  intrinsic  IR  frequency  does  notshift  appreciably.  Instead,  only  the  relative  contributions  of highly overlapped adjacent bands are changing."
https://doi.org/10.1366/000370210792434396

# Results
Psi4 produces acetone vibrations across 4 modes, initially these are subtracted from 10 varying concentrations of water and acetone and used as the target using a GNN. These results show variation of +/- 5cm-1 from the calculated difference.

![predictions from basic GNN](./predictions_from_basic_GNN.png)

The bond length error (O–H) stabilized around 0.4 Å, with early fluctuation likely due to KL warm-up. Angle error (H–O–H) dropped from over 60° to under 10°, showing the model learning the getting close to the correct water angle. Training and validation losses both decreased steeply and then plateaued, indicating learning without overfitting.

![training and validation loss](./training_and_validation_loss.png)

![angle error](./angle_error.png)

![bond error](./bond_length_error.png)

The trained Conditional GraphVAE model was evaluated for its ability to reconstruct geometries. The focus was on maintaining realistic O–H bond lengths (~0.96 Å) and the H–O–H bond angle (~104.5°) across different concentration conditions.

O–H Bond Lengths
The predicted bond lengths for O–H₁ and O–H₂ remain close to the target value in most concentration settings, but certain indices show elevated bond lengths (~2.0 Å), indicating occasional model instability. A dashed line at 0.96 Å provides a reference for the ideal value. These deviations suggest a need to strengthen the bond loss term or explore outlier geometries in training.

H–O–H Bond Angle 
The model is less accurate with angle prediction. While the target angle is ~104.5°, fluctuations are seen across concentration indices, with angles as low as ~35° and as high as ~195°. 

[predict OH bond angle](./predict_OH_bond_angle)

[predict OH bond length](./predict_OH_bond_length)

# References

https://cs229.stanford.edu/proj2017/final-reports/5244394.pdf

https://www.spectroscopyonline.com/view/five-reasons-why-not-every-peak-shift-in-infrared-ir-spectra-indicates-a-chemical-structure-change








