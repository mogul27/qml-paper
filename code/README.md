##### Envrionment Setup
This program has been tested on Python 3.11.6. It is recommended to create a Python virtual environment to run the Python files and notebooks in this project. Note that the kernel will need to be changed in the notebook in order to run using the virtual environment.

To install the required Python packages, use the included requirements.txt file:

*pip3 install -r requirements.txt*
or
*pip install -r requirements.txt*

##### Input Data

The data to run the classification is included in the data folder. To re-create the files in the data folder, open a jupyter notebook within the top level code directory:

*jupyter notebook*

Run all cells in the data_preprocess.ipynb notebook and this will re-create the 4 datasets from the UCI ML repository in the data folder. Running all cells in the synthetic_data_generation.ipynb notebook will re-create the 3 synthetic datasets. The synthetic datasets are generated using the make_blobs and make_classification functions available in sklearn. Links to these  an be found in the references section of this README file.

##### Running a processing pipeline

The changeable parameters for running a set of classifcations can be found in the run_pipeline.py Python file under the start of the main file section (if __name__=="__main__":). The datasets used for classification and the Pauli feature maps used for constructing the QSVM models can also be changed in this section. Do not change the simulator parameter of the run_pipeline function as this requires additional setup. Only quantum simulators were used in this project.

The description of how the parameters impact the QSVM classification pipeline are found in the docstring parameter description of the function run_pipeline in the run_pipeline.py Python file.

To classify the selected datasets with QSVM models, run the run_pipeline.py Python file from the top level code directory:

*python3 run_pipeline.py*
or
*python run_pipeline.py*

Results from running a processing pipeline will be output to the results folder. Accuracies are output as an array where the first element is the train set accuracy and the second element is the test set accuracy. The settings for that run are also output to the same file.

##### Re-running regression analysis

The notebooks used to obtain the regression analysis described in the dissertation document are located in the top-level of the code directory. These files are named pca_regression.ipynb and umap_regression.ipynb. These files leverage the results spreadsheets stored in the assets folder. To re-run the regression analysis, open a jupyter notebook within the top level code directory:

*jupyter notebook*

Run all cells in the pca_regression.ipynb and umap_regression.ipynb to obtain the regression results. The regressions are run using the statsmodels OLS function referenced in this README file.

### References

##### Datasets used

Alpaydin, E., Kanyak, C.K. 1998. *Optical Recognition of Handwritten Digits* \[Online]. UCI Machine Learning Repository. Available from: https://doi.org/10.24432/C50P49 [Accessed 3 December 2023].

Blackard, J., 1998. *Covertype* \[Online]. UCI Machine Learning Repository. Available from: https://doi.org/10.24432/C50K5N [Accessed 25 December 2023].

Bohanec, M., 1988. *Car Evaluation* \[Online]. UCI Machine Learning Repository. Available from: https://doi.org/10.24432/C5JP48 [Accessed 27 June 2023].

The Audubon Society Field Guide to North American Mushrooms. 1987. *Mushroom* \[Online]. UCI Machine Learning Repository. https://doi.org/10.24432/C5959T. [Accessed 20 July 2023].

##### Key packages

Scikit-Learn documentation. 2024a [Online]. Available From: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs [Accessed 30 March 2024]

Scikit-Learn documentation. 2024b [Online]. Available From: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification [Accessed 30 March 2024]

Statsmodels documentation. 2024 [Online]. Available From https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html


