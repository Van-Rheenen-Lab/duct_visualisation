# Mammary duct simulation, experimental tracing & annotation and dendrogram visualisation package
A package developed towards understanding the dynamics of mammary stem cells during puberty & adulthood. The repository contains:

- stochastic simulations of clonal propagation during pubertal growth and adult maintenance
- a GUI for 2D and 3D image tracing and annotation
- visualisation utilities for experimental and simulated duct networks, including dendrograms and clone overlays

## Aims

We aim to connect experimentally observed ductal trees and clonal patterns to simple, testable models of stem cell behavior. Simulations generate trees and clone distributions. The GUI streamlines consistent tracing and annotation. Visualisation tools aim to create consistent representations of simulated and experimental data.

## System requirements
Supported: any system that can run python 3.9 to 3.12
Tested: Windows 10, Windows 11 and macOS 13. Tested on python 3.10 to 3.12

No special hardware required.
Recommended: 8 GB RAM minimum, more for large 3D stacks.

Dependencies

The exact list is defined in pyproject.toml. Packages:

- numpy
- scipy
- matplotlib
- tifffile
- scikit-image
- scikit-learn
- pyqt5
- opencv-python-headless
- pandas
- seaborn
- networkx
- shapely

If you plan to build a standalone executable for the GUI:
- pyinstaller
- imagecodecs


## Installation

After cloning, from python terminal at the location of the package:

```
pip install -e .
```

This should take not more than a couple of minutes.

# Simulation

The package provides abstract, experimentally informed simulations of mammary duct dynamics in two regimes. Outputs are ductal trees with per edge annotations that can be visualised next to traced glands.

### Pubertal growth

A branching process grows a tree from active tips. At each step a tip can extend, branch, or terminate. Clonal labels travel with growing tips and are written to the resulting segments. The model is designed to capture large scale features of pubertal outgrowth rather than exact geometry.

<img width="1000" height="500" alt="puberty_ductal_tree" src="https://github.com/user-attachments/assets/50640760-bc93-4d9d-a122-aa708455430c" />
<img width="500" height="464" alt="image" src="https://github.com/user-attachments/assets/adb561a0-e367-45f2-adea-fe5cf460d3b8" />

### Adult maintenance

On an established ductal network produced by the puberty simulation, adulthood is modeled. Local loss and replacement events maintain the tissue over time. Turnover is modeled as neutral currently.

Visualisation of competition in a single duct:

<img width="500" height="500" alt="pubertal_clones_single_duct" src="https://github.com/user-attachments/assets/9eef650a-917d-49c8-a6d1-13e01e0960ec" />

### Demo
The scripts simulation/development/puberty_population_dynamics.py and simulation/development/testing_adulthood.py run out of the box and show a full simulation. They should plot dynamics of the simulated glands. On a normal desktop computer these simulations will take a minute to run.



# Image annotation tool
The image annotation tool can be run from python directly, or a .exe can be built.
first install imagecodecs and pyinstaller:

```
pip install pyinstaller imagecodecs
```
Add the hooks for imagecodex in a folder called hooks. build using the command: 
```
pyinstaller --onefile --additional-hooks-dir=hooks --hidden-import=imagecodecs duct_tracking_GUI.py
```

![1](https://github.com/user-attachments/assets/ec626eb2-a4c6-4977-8ac9-10d8d186856a)

### Demo
The tool should run from python out of the box. It does not require an image, so a tree can be made even without actual data. For in-depth instructions, press "Instructions" in the window.


# Simulation & Visualisation tools
The rest of the software has been built specificially to visualise the neccesary aspects of our own imaging data, and as such, is in a less flexible state. Work will be done to increase the wider usability of this code. 

### Demo
A quick demo can be achieved by making a dendrogram in the duct_tracing_GUI, and building its tree in analysis/simple_dendrogram_from_tracing.py. An example dendrogram has also been made, so running simple_dendrogram_from_tracing.py should work out of the box. On a simple desktop computer this should run almost instantly.

Visualisation of traced ducts of experimental 3d image:
![2](https://github.com/user-attachments/assets/71a404ab-4edc-429c-b3e6-557c5e319112)
And the dendrogram abstraction of an annotated 3d image:
![3](https://github.com/user-attachments/assets/e8507a29-7120-40aa-bf59-1a6801ddfbcc)
