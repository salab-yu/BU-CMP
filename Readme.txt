BU-CMP: README

This code, Bayesian Updating for Cementitious Material Properties (BU-CMP), was written by Dr. Md Samdani Azad under the guidance of Professor Tong-Seok Han at Yonsei University in Seoul, Republic of Korea. Their expertise and support were key in shaping this work, which applies 
Bayesian updating to mechanical property evaluation of cement paste.


Instruction Manual for Running the Bayesian updating Script:

Overview
- This script performs Bayesian analysis to calculate the mechanical propety of cement paste and the concept is based on conjugate prior.

- A CSV file is prepared to give the inputs: prior dataset and observed dataset.
- High-quality figures of prior and posterior distributions, customizable in terms of font, titles, and axis range.
- Figures can be saved in multiple formats, including PDF, SVG, PNG, JPG, and TIFF.

System Requirements
- Python Version: Ensure Python --------.
- Anaconda Environment: Install and use the script within an Anaconda environment for optimal compatibility.
- Required Libraries:
  numpy
  scipy
  matplotlib
  pandas
  seaborn

Setup Instructions
- Download the Script: Save the main Python script (Bayesian_updating.py) to a desired directory.
- Prepare the Input CSV File.
- Ensure the input CSV file is correctly formatted as required by the script (e.g., column names, data types).
- Place the input file in the same directory as the script or provide the full file path.
- Install Dependencies: Open the Anaconda Prompt and install necessary libraries.

How to use input file
- The input file is in the csv format.
- Each row has two data seperated by a comma. 
- The first data is prior data and second data is observed data.
- If there is no first data, keep blank and put comma and put second data.
- Follow the example below.
   prior_data,observed_data
   24.47,19.12
   20.89,18.87
   19.48,19.40
   15.34,18.86
   ,18.84
   ,18.65
   ,19.46
   ,18.69

Running the Script
- type the following in Anaconda prompt or command prompt in MS Windows: cd path\to\your\directory
- then type as follows to run the code: python Bayesian_updating.py
