# Binomial Options Pricing Model Playground: Understanding Discrete-Time Option Pricing

## 1. What Is This?

This interactive application demonstrates the **Binomial Options Pricing Model**, a discrete-time framework for pricing options by constructing a price tree (lattice).

- **What It Is:**  
  The model simulates possible price paths for an underlying asset by dividing the time to expiration into a finite number of steps. At each step, the asset price can move up or down by predetermined factors, which helps in computing the option's fair value under risk-neutral pricing.

- **Why Teach It:**  
  It simplifies the intuition behind risk-neutral pricing and serves as a foundation for understanding more advanced continuous-time models like Black-Scholes. In addition, it illustrates how delta hedging can be implemented in a discrete setting.

- **Example:**  
  Price a European call (or put) option using a 2-step binomial tree, and observe the corresponding delta hedging ratio.

**Note:** This tool is for educational purposes only. Accuracy is not guaranteed, and the computed option prices and hedging ratios do not represent actual market values. The author is Luís Simões da Cunha.

## 2. Setting Up a Local Development Environment

### 2.1 Prerequisites

1. **A computer** (Windows, macOS, or Linux).
2. **Python 3.9 or higher** (Python 3.12 preferred, but anything 3.9+ should work).  
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
   git clone https://github.com/yourusername/binomial_options_pricing_playground.git
   ```
4. Enter the project folder:
   ```bash
   cd binomial_options_pricing_playground
   ```

#### Option 2: Download as ZIP

1. Visit [https://github.com/yourusername/binomial_options_pricing_playground](https://github.com/yourusername/binomial_options_pricing_playground)
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
- **NumPy** (for numerical computations)
- **Matplotlib** (for plotting results)

## 3. Running the Application

To launch the Binomial Options Pricing Model playground, execute:

```bash
streamlit run binomial_options_pricing.py
```

This should open a new tab in your web browser with the interactive tool. If it does not open automatically, check the terminal for a URL (e.g., `http://localhost:8501`) and open it manually.

### 3.1 Troubleshooting

- **ModuleNotFoundError:** Ensure the virtual environment is activated (`venv\Scripts\activate` on Windows or `source venv/bin/activate` on macOS/Linux).
- **Python not recognized:** Make sure Python is installed and added to your system's PATH.
- **Browser does not open automatically:** Manually enter the `http://localhost:8501` URL in your browser.

## 4. Editing the Code

If you want to make modifications:
1. Open `binomial_options_pricing.py` in **VS Code**.
2. Modify the code as needed.
3. Restart the Streamlit app after changes (press `Ctrl + C` in the terminal to stop, then rerun `streamlit run binomial_options_pricing.py`).

## 5. Additional Resources

- **Streamlit Documentation:** [docs.streamlit.io](https://docs.streamlit.io)
- **Binomial Options Pricing Model Overview:** [Investopedia Guide](https://www.investopedia.com/terms/b/binomialoptionpricingmodel.asp)
- **Risk-Neutral Pricing and Delta Hedging:** Look for educational materials and tutorials online.

## 6. Support

For issues or suggestions, open an **Issue** on GitHub:  
[https://github.com/yourusername/binomial_options_pricing_playground/issues](https://github.com/yourusername/binomial_options_pricing_playground/issues)

---

*Happy exploring the Binomial Options Pricing Model and enhancing your understanding of discrete-time option pricing!*
