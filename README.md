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

3. Then run using:
   ```bash
   python app.py
   ```

 Note: Comment out any `plt.show()` if too many graphs open, or run cell-by-cell.

---

###  Method 3: Run Online via Streamlit Community Cloud

Even though this is a notebook-based ML project, you can still run it online using Streamlit Cloud:

1. Go to: [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Create a new app:
   - Repo: House Price Predictor
   - Branch: main
   - File: `House Price Prediction.ipynb`
4. Click "Deploy"

Note: This won't create a Streamlit UI — just an online way to execute or share your project.


##  Made With
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, sklearn
-  Notepad (for README and requirements)


##  Author

This project is created by a Mohd Farhan, Computer Science student as part of learning and practice.
