# capstone-covid
Exploring COVID simulation data using AI

## Introduction
The SpikeAnalytics project's goal is to leverage machine learning to mine COVID spike simulation data for insights. The analysis pipeline consists of two steps: feature extraction and the SpikeAnalytics dashboard. In the first step, molecular dynamics trajectory files are analyzed and features are extracted on a frame-by-frame basis. These feature sets are then used in the SpikeAnalytics dashboard to train an SGD classifier to predict whether the spike protein is open or closed in a given frame. The dashboard allows users to view not only the results of the model but also the details of the important features. 

The github can be found at: https://github.com/mkmontgo46/capstone-covid

## Technical Requirements
Need at least 32 GB of RAM

## Data Preparation
The SpikeAnalytics pipeline takes as input simulation data in folders with the word "TRAJECTORIES" in them. The trajectories themselves should be in the form of .dcd files, and a .psf file should also be included. The feature extraction will need to write .csvs to a "results" subfolder within the data directory. Datasets which feature simulations of a closed spike should be labeled as such in the TRAJECTORIES folder name. For example:

  ./Spike_Dataset/TRAJECTORIES_open/
  
    open_trajectory_1.dcd
    
    open_trajectory_2.dcd
    
    open_trajectory_3.dcd
    
    open.psf
    
    results/
    
  ./Spike_Dataset/TRAJECTORIES_closed/
  
    closed_trajectory_1.dcd
    
    closed_trajectory_2.dcd
    
    closed_trajectory_3.dcd
    
    closed.psf
    
    results/
    
    
Finally the input data directory (in the above case, "./Spike_Dataset/") needs to be saved as an environment variable named SPIKEDATASET_DIR. Datasets such as these can be downloaded from https://amarolab.ucsd.edu/covid19.php

## Local Installation
Install miniconda for Linux. 
Create the conda environment:
  ```
  conda env create -f environment.yml
  ```
Activate the environment:
  ```
  conda activate env1
  ```
Create the environment variable:
  ```
  export SPIKEDATASET_DIR=[data directory]
  ```
Run feature extraction:
  ```
  python dcd_featureExtract.py -dcd [dcd file] -psf [psf file] -outdir [results subfolder]
  ```
Run dashboard:
  ```
  python spike_dashboard.py
  ```

## Docker
Build:
```
docker build -t amarolab/spike .
```

Run:
```
docker run -v $(pwd):/workdir -i -t -p 8888:8888 amarolab/spike
```

Start jupyter notebook from container:
```
jupyter notebook --ip='*' --port=8888 --no-browser --allow-root
```
You can then view the Jupyter Notebook by opening the url that was generated by the previous command.

## Using the Dashboard
The SpikeAnalytics dashboard is designed to work from top to bottom to help the user progress through the data analysis process. The user starts by selecting at least two datasets to analyze from the first drop-down list, labeled **Select Profiled Data-sets to Analyze**. At least one open and at least one closed dataset need to be included in this selection. At this point, the **Preview Model Features** button will be enabled, but the user may continue to adjust their model features using the following three inputs:

  * **Select Features to Use**: This drop-down menu allows the user to select the types of features they would like to be included in the model. Each feature type has been calculated in the feature extraction step for all substructures, including the receptor binding domain, or RBD (with the exception of RBD distances), the central helix, or CH, the backbone, and all 70 glycans. The types of features are:
    * RBD Distances: the distance from the receptor-binding domain (RBD) to the substructure
    * RMSD: the root-mean squared distance of the substructure
    * Radius of Gyration: the radius of gyration of the substructure
    * x location: the x-coordinate of the center of mass of the substructure
    * y location: the y-coordinate of the center of mass of the substructure 
    * z location: the z-coordinate of the center of mass of the substructure
    
    The default selection is to use all feature types.
 * **Set Max RBD Neighborhood**: This numerical input allows the user to set a limit on how far away from the receptor binding domain (RBD) a substructure can be for its features to be included in the model. For example, if the max neighborhood is set to 6 nm, none of the features for a glycan whose center of mass is 7 nm away from the center of mass fo the RBD will be used in the model. The default value is 4 nm.
 * **Set Max Feature Correlation**: This numerical input allows the user to set a limit on how well-correlated two features can be. For example, if the max correlation is set to 0.4, two features that have a correlation coefficient of 0.6 will be excluded from the model. The default value is 0.5.

Once the input features have been selected, the user can preview these features by clicking the button labeled **Preview Model Features**. This will trigger the loading and processing of the feature sets, and histograms of the features will be plotted, with separate figures for open and closed data. For readability purposes, only the features for a single feature type will be plotted. The default is that the first feature type selected will be plotted initially, but the user can change which feature type they are viewing by making a selection in the drop-down labeled **View**. 

Once the user is comfortable with the features they are inputting, they can train the model by clicking on the button labeled **Train Model** (the button will now be enabled). This will trigger the training and testing of a Stochastic Gradient Descent (SGD) classifier using python's sci-kit learn toolbox. The testing performance (precision and recall) for the classifier will be printed on the dashboard. Additionally, the feature importances for the top-10 features will be plotted in a bar graph, with the most important features shown on the left. The chains of the substructures the features were derived from will be encoded in the colors of the bars.

Below the feature importances graph, the user can view information for a single feature. The value of that feature over time (aka frames) for each individual trajectory used in the model will be plotted on the left. On the right, histograms for all data used in the model are included for that feature. Open data is color-coded blue, and closed data is color-coded red. The default is for the most important feature to be plotted, but the user can click on a bar in the feature importances graph to change which feature is shown. The figure will be labeled with the feature that is plotted.

Finally, the user can view the substructures corresponding to the top features in 3D in the context of the overall spike protein by clicking on the button labeled **Show Features in 3D**. This will trigger the loading and processing of a single trajectory file for both the open and closed data. Then a 3D scatter plot of the closed spike will be shown on the left, and a 3D scatter plot of the open spike will be shown on the right. The backbone, sidechains, and all glycans will be plotted with small gray markers, the RBD will be plotted with small blue markers, the central helix will be plotted with small red markers, and the top 5 most important glycans in the classifier will be plotted with large markers in various colors. The user can toggle substructures on and off using the legends, and can zoom, pan, and rotate the figures in the dashboard. Any camera change will be applied to both figures, so the user can view the substructures for both open and closed data side by side.
