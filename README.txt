##File Structure
UNET.py                     				# 3D UNet backbone (encoder, decoder, expert/tower branches)
MMOE.py                     				# MMoE multi-task network definition
GATE.py                     				# Gating network for expert weighting
Constants.py              		  		# Global paths, parameters, and training settings
main_MMOE.py                			# Main training and testing pipeline
loss_calculation.py         			# Loss functions
traindata_and_testdata_maker.py  	# Training/test data loading and preprocessing
segy.py                    	 			# SEGY seismic data I/O
README.md                   			# This document

##Dependencies
Python >= 3.12
PyTorch >= 2.4.0
CUDA >= 12.4

##Dataset
This project uses the public F3 seismic dataset (https://terranubis.com/datainfo/F3-Demo-2023).
Seismic data: SEGY format
Well data: CSV (V, DEN, Z) + Excel (well locations)
Users must download the dataset independently and set paths in Constants.py.

##How to Run
1. Set paths and parameters in Constants.py.
2. Run the main script main_MMOE.py.
3. The script automatically:
	Loads and preprocesses seismic and well data
	Loads and preprocesses training/test samples
	Trains the MMOE inversion network and U-Net forward network
	Outputs loss and accuracy
	Saves trained network models