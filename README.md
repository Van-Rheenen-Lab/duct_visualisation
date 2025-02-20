# Duct_tracker, simulation & Dendrogram visualisation

This repository was build to support the paper:

"Puberty as a Superspreader of Mutations Driving Tumor Risk in Breast Tissue, H. Hristova et al. 2025". 

It contains a GUI to conveniently trace & annotate 2D- and 3D- mammary gland images, multiple methods for visualising these, methods to analyse fluorescent data along the dendrograms, and simulations of stem cell dynamics in mammary gland pubertal development and adult maintainance. 

# Image annotation tool
The image annotation tool can be run from python directly, or a .exe can be built.
build using the command: `pyinstaller --onefile --additional-hooks-dir=hooks --hidden-import=imagecodecs duct_tracking_GUI.py` We can also provide the .exe on request.

# Simulation & Visualisation tools
The rest of the software has been built specificially to visualise the neccesary aspects of the data for the publication, and as such, is in a less flexible state. Work will be done to increase the wider usability of this code.

![image](https://github.com/user-attachments/assets/3aa8698a-a1f8-4127-9a2b-d627fa3bb546)
![image](https://github.com/user-attachments/assets/ff0a3e95-ccca-4184-a893-ca9a6f8bd891)
