# Duct_tracker, simulation & Dendrogram visualisation

This repository contains a GUI to conveniently trace & annotate 2D- and 3D- mammary gland images, multiple methods for visualising these, methods to analyse fluorescent data along the dendrograms, and simulations of stem cell dynamics in mammary gland pubertal development and adult maintainance. 

# Image annotation tool
The image annotation tool can be run from python directly, or a .exe can be built.
build using the command: `pyinstaller --onefile --additional-hooks-dir=hooks --hidden-import=imagecodecs duct_tracking_GUI.py` We can also provide the .exe on request.

![1](https://github.com/user-attachments/assets/ec626eb2-a4c6-4977-8ac9-10d8d186856a)



# Simulation & Visualisation tools
The rest of the software has been built specificially to visualise the neccesary aspects of our own imaging data, and as such, is in a less flexible state. Work will be done to increase the wider usability of this code.

![2](https://github.com/user-attachments/assets/71a404ab-4edc-429c-b3e6-557c5e319112)
![3](https://github.com/user-attachments/assets/e8507a29-7120-40aa-bf59-1a6801ddfbcc)
