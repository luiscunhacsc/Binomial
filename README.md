# VaR & ES Playground: Understanding Downside Risk

## 1. What Is This?

This interactive application demonstrates two critical risk management measures:

- **Value at Risk (VaR):**  
  Estimates the maximum expected loss over a specified time period at a given confidence level. For example, a 1-day 95% VaR of €50,000 implies that there is a 5% chance the portfolio could lose more than €50,000 in one day.

- **Expected Shortfall (ES):**  
  Also known as Conditional VaR (CVaR), it calculates the average loss beyond the VaR threshold, giving a clearer picture of tail risk.

The playground offers two methods for calculating these metrics:
- **Parametric (Variance-Covariance) Method:** Assumes normally distributed returns and uses the mean and volatility to compute VaR and ES.
- **Historical Simulation Method:** Computes risk measures using simulated (or historical) returns without assuming a specific distribution.

With interactive controls, you can adjust parameters such as portfolio value, expected daily return, daily volatility, confidence level, and the historical window length to see how these factors affect your risk estimates.

## 2. Setting Up a Local Development Environment

### 2.1 Prerequisites

1. **A computer** (Windows, macOS, or Linux).
2. **Python 3.9 or higher** (Python 3.12 preferred, but anything 3.9+ should be fine).  
   - If you do not have Python installed, visit [python.org/downloads](https://www.python.org/downloads/) to install the latest version.
3. **Visual Studio Code (VS Code)**
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)
4. **Git** (optional, but recommended for cloning the repository).  
   - Install from [git-scm.com/downloads](https://git-scm.com/downloads)

### 2.2 Downloading the Project

#### Option 1: Cloning via Git (Recommended)

1. Open **Terminal** (macOS/Linux) or **Command Prompt** / **PowerShell** (Windows).
2. Navigate to the folder where you want to download the project:
   ```bash
   cd Documents
   ```
3. Run the following command:
   ```bash
   git clone https://github.com/yourusername/var_es_playground.git
   ```
4. Enter the project folder:
   ```bash
   cd var_es_playground
   ```

#### Option 2: Download as ZIP

1. Visit [https://github.com/yourusername/var_es_playground](https://github.com/yourusername/var_es_playground)
2. Click **Code > Download ZIP**.
3. Extract the ZIP file into a local folder.

### 2.3 Creating a Virtual Environment

It is recommended to use a virtual environment (`venv`) to manage dependencies:

1. Open **VS Code** and navigate to the project folder.
2. Open the integrated terminal (`Ctrl + ~` in VS Code or via `Terminal > New Terminal`).
3. Run the following commands to create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

### 2.4 Installing Dependencies

After activating the virtual environment, install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This command installs libraries such as:
- **Streamlit** (for the interactive UI)
- **NumPy** and **SciPy** (for mathematical and statistical computations)
- **Matplotlib** (for plotting results)

## 3. Running the Application

To launch the VaR & ES playground, execute:

```bash
streamlit run var_es_playground.py
```

This should open a new tab in your web browser with the interactive tool. If it does not open automatically, check the terminal for a URL (e.g., `http://localhost:8501`) and open it manually.

### 3.1 Troubleshooting

- **ModuleNotFoundError:** Ensure the virtual environment is activated (`venv\Scripts\activate` or `source venv/bin/activate`).
- **Python not recognized:** Ensure Python is installed and added to your system's PATH.
- **Browser does not open automatically:** Manually enter the `http://localhost:8501` URL in your browser.

## 4. Editing the Code

If you want to make modifications:
1. Open `var_es_playground.py` in **VS Code**.
2. Modify the code as needed.
3. Restart the Streamlit app after changes (`Ctrl + C` to stop, then rerun `streamlit run var_es_playground.py`).

## 5. Additional Resources

- **Streamlit Documentation:** [docs.streamlit.io](https://docs.streamlit.io)
- **Value at Risk (VaR) Overview:** [Investopedia Guide](https://www.investopedia.com/terms/v/var.asp)
- **Expected Shortfall (ES) Information:** [Investopedia Guide](https://www.investopedia.com/terms/e/expectedshortfall.asp)

## 6. Support

For issues or suggestions, open an **Issue** on GitHub:  
[https://github.com/yourusername/var_es_playground/issues](https://github.com/yourusername/var_es_playground/issues)

---

*Happy exploring VaR & ES and managing downside risk!*
