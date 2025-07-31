#  House Price Prediction

This is a simple machine learning project made by a student to predict house prices using models like Linear Regression, Decision Tree, and Random Forest. It is trained on a housing dataset.


##  Files Included

- `House Price Prediction.ipynb` → Main notebook file
- `housing.csv` → Dataset used for training and testing
- `README.md` → This file tells how to run program
- `requirements.txt` → List of Python libraries used


##  How to Run This Project

###  Method 1: Using Jupyter Notebook (recommended)

1. Download or clone this repo:
   ```bash
   git clone https://github.com/mohdfarhann/House_Price_predictor.git
   cd House_price_predictor
   ```

2. Open `House Price Prediction.ipynb` in Jupyter Notebook or VS Code.

3. Make sure `housing.csv` is in the same folder.

4. Run all cells and see the results.

5. you will see output of custom input and by giving inputs you will see output of you inputs.
---

###  Method 2: Run as Python Script (app.py)

> If you prefer running as a `.py` file instead of notebook:

1. Convert notebook to Python:
   Run this line in your Terminal
   jupyter nbconvert --to script "House Price Prediction.ipynb"
   

2. It will generate a file like `House Price Prediction.py`.  
   You can also rename it to `app.py`.
3. Or run the .py file given
4. make sure all libraries were installed mentioned in requirements.txt
5. Then run using:
   ```bash
   python app.py
   ```
 Note: Comment out any `plt.show()` if too many graphs open, or run cell-by-cell.


## Future Plan 
 Note: A Streamlit-based interactive web app version is under development and will be added soon!


##  Made With
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, sklearn
-  Notepad (for README and requirements)


##  Author

This project is created by a Mohd Farhan, Computer Science student as part of learning and practice.
