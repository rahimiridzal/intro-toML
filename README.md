# intro-toML
Introduction to Machine Learning classification problem. Includes Data Preprocessing and Decision Tree implementation using only NumPy.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Project Description

This project provides an end-to-end overview of how to approach a simple classification problem. The aim is to offer an alternative solution that intentionally avoids the use of Pandas, as there are already numerous open resources available utilising it.

Note: While this project intentionally avoids Pandas, it does not imply that Pandas is ineffective or should be avoided in general. It simply provides an alternative perspective to broaden your skill set and encourage exploration.

## Installation

To set up the project environment, follow these steps:

1. Clone the project repository:

    ```shell
    $ git clone https://github.com/rahimiridzal/intro-toML.git
    $ cd repository

2. Create a new conda environment using the provided 'environment.yml' file:
    
    ```shell
    $ conda env create -f environment.yml

    #This command will create a new conda environment with all the necessary dependencies specified in the environment.yml file.

3. Activate the newly created environment:
    ```shell
    $ conda activate intro-toML

    #Replace intro-toML with the name of the environment if you specified a different name in the environment.yml file.

4. Launch jupyter notebook:
    ```shell
    $ jupyter notebook

5. In the Jupyter Notebook interface, navigate to the project directory and open the notebook file 'intro-toML.ipynb'.

You are now ready to use the project within the configured conda environment. Ensure that the environment is activated whenever you want to run the project.
    ```shell
    $ conda activate intro-toML
    $ jupyter notebook