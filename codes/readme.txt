Code for AISTATS 2025 submission:
Distribution-Informed Prediction via Kernelized Stein Discrepancy Calibration

%%%%%%%%%%%%%%%% Preview %%%%%%%%%%%%%%%%
There are 2 python file, 4 jupyter notebooks in this folder.

%%%%%%%%%%%%%%%% Environment %%%%%%%%%%%%%%%%
All programs were run under Python 3.11.5.
To download the python libraries: pip install -r requirements.txt

%%%%%%%%%%%%%%%% Data %%%%%%%%%%%%%%%%
The data should be put in the /data/ folder.
For CESM dataset, you could find it here: https://figshare.com/collections/Large_ensemble_pCO2_testbed/4568555

%%%%%%%%%%%%%%%% Code structure %%%%%%%%%%%%%%%%
model.py is used for defining FFN

utils.py is used for loading CESM dataset

toy_example_1.ipynb is used to reproduce all the results in toy example 1, including Table 1, S1, S2 and Figure 3.

toy_example_2.ipynb is used to reproduce all the results in toy example 2, including Table S3, S4 and Figure 4.

pco2_result.ipynb is used to reproduce all the results in Application to air-sea CO2 flux, including Table 2, 3, S5, S6, S7, S8 and Figure 1, 5, S1, S2.

%%%%%%%%%%%%%%%% Running the code %%%%%%%%%%%%%%%%

toy_example_1.ipynb and toy_example_2.ipynb can run directly.

Please run pco2_train.ipynb to generate necessary data before running pco2_result.ipynb.