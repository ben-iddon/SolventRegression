# SolventRegression
A tool for regression analysis of solvent effects on chemical processes.

Run the python script with solvent_descriptors.csv in the same directory

The input file should contain thee columns : Solvent ID, Solvent Name, log k
The only other compulsory input is the value for deltaN

**Example usage (using the example files provided):**

python solvent_regression.py example_input.csv 0
will run the regression with deltaN = 0, and output results into console

python solvent_regression.py example_input.csv 0 -o
will run the regression with deltaN = 0, and write detailed regression results into a file output.csv 
