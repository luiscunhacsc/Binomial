import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#######################################
# 1) Callback functions to reset or set lab parameters
#######################################
def reset_parameters():
    st.session_state["S_slider"] = 100.0
    st.session_state["K_slider"] = 100.0
    st.session_state["T_slider"] = 1.0
    st.session_state["r_slider"] = 0.05
    st.session_state["sigma_slider"] = 0.2
    st.session_state["steps_slider"] = 2
    st.session_state["option_type_radio"] = 'call'

def set_lab1_parameters():
    # Lab 1: Basic 2-step tree for a European Call
    st.session_state["S_slider"] = 100.0
    st.session_state["K_slider"] = 100.0
    st.session_state["T_slider"] = 1.0
    st.session_state["r_slider"] = 0.05
    st.session_state["sigma_slider"] = 0.2
    st.session_state["steps_slider"] = 2
    st.session_state["option_type_radio"] = 'call'

def set_lab2_parameters():
    # Lab 2: Increase steps to see convergence toward Black-Scholes
    st.session_state["S_slider"] = 100.0
    st.session_state["K_slider"] = 100.0
    st.session_state["T_slider"] = 1.0
    st.session_state["r_slider"] = 0.05
    st.session_state["sigma_slider"] = 0.2
    st.session_state["steps_slider"] = 50
    st.session_state["option_type_radio"] = 'call'

def set_lab3_parameters():
    # Lab 3: Change option type to Put and observe differences
    st.session_state["S_slider"] = 100.0
    st.session_state["K_slider"] = 100.0
    st.session_state["T_slider"] = 1.0
    st.session_state["r_slider"] = 0.05
    st.session_state["sigma_slider"] = 0.2
    st.session_state["steps_slider"] = 2
    st.session_state["option_type_radio"] = 'put'

#######################################
# 2) Binomial Options Pricing function with delta calculation
#######################################
def binomial_option_pricing(S, K, T, r, sigma, steps, option_type='call'):
    """
    Prices a European option using a multi-step binomial tree.
    Returns (price, delta) where delta is approximated using the first step.
    """
    delta_t = T / steps
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    p = (np.exp(r * delta_t) - d) / (u - d)
    
    # Initialize option values at maturity:
    option_values = np.zeros(steps + 1)
    for i in range(steps + 1):
        stock_price = S * (u ** i) * (d ** (steps - i))
        if option_type == 'call':
            option_values[i] = max(stock_price - K, 0)
        else:
            option_values[i] = max(K - stock_price, 0)
    
    # Backward induction through the tree:
    for j in range(steps, 0, -1):
        for i in range(j):
            option_values[i] = np.exp(-r * delta_t) * (p * option_values[i + 1] + (1 - p) * option_values[i])
    price = option_values[0]
    
    # Compute delta using the first step nodes:
    if steps == 1:
        # One-step tree: directly compute payoffs
        S_up = S * u
        S_down = S * d
        if option_type == 'call':
            V_up = max(S_up - K, 0)
            V_down = max(S_down - K, 0)
        else:
            V_up = max(K - S_up, 0)
            V_down = max(K - S_down, 0)
        delta = (V_up - V_down) / (S_up - S_down)
    elif steps == 2:
        # Two-step tree: manually compute time-1 values
        # Prices at maturity:
        S_uu = S * u * u
        S_ud = S * u * d
        S_dd = S * d * d
        if option_type == 'call':
            V_uu = max(S_uu - K, 0)
            V_ud = max(S_ud - K, 0)
            V_dd = max(S_dd - K, 0)
        else:
            V_uu = max(K - S_uu, 0)
            V_ud = max(K - S_ud, 0)
            V_dd = max(K - S_dd, 0)
        # Backward induction for time-1 nodes:
        V_u = np.exp(-r * delta_t) * (p * V_uu + (1 - p) * V_ud)
        V_d = np.exp(-r * delta_t) * (p * V_ud + (1 - p) * V_dd)
        delta = (V_u - V_d) / (S * u - S * d)
    else:
        # For steps > 2, approximate delta via a one-step difference
        V_up = binomial_option_pricing(S * u, K, T - delta_t, r, sigma, steps - 1, option_type)[0]
        V_down = binomial_option_pricing(S * d, K, T - delta_t, r, sigma, steps - 1, option_type)[0]
        delta = (V_up - V_down) / (S * u - S * d)
    
    return price, delta

#######################################
# 3) Configure the Streamlit app layout and sidebar
#######################################
st.set_page_config(layout="wide")
st.title("📊 Binomial Options Pricing Model Playground")
st.markdown("Explore how a discrete-time binomial tree can be used to price European options and understand delta hedging.")

with st.sidebar:
    st.header("⚙️ Parameters")
    st.button("↺ Reset Parameters", on_click=reset_parameters)
    
    st.markdown("### Option Parameters")
    S = st.slider("Current Stock Price (S)", 50.0, 150.0, 100.0, key='S_slider')
    K = st.slider("Strike Price (K)", 50.0, 150.0, 100.0, key='K_slider')
    T = st.slider("Time to Maturity (years)", 0.1, 5.0, 1.0, key='T_slider')
    r = st.slider("Risk-Free Interest Rate (r)", 0.0, 0.2, 0.05, key='r_slider', format="%.2f")
    sigma = st.slider("Volatility (σ)", 0.1, 1.0, 0.2, key='sigma_slider', format="%.2f")
    steps = st.slider("Number of Steps", 1, 100, 2, key='steps_slider')
    option_type = st.radio("Option Type", ["call", "put"], key='option_type_radio')
    
    st.markdown("---")
    st.markdown(
    """
    **Disclaimer**  
    *This tool is for educational purposes only. Accuracy is not guaranteed, and the computed option prices and deltas do not represent actual market values. The author is Luís Simões da Cunha.*
    """)
    
    st.markdown("""
    <div style="margin-top: 20px;">
        <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en" target="_blank">
            <img src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" alt="CC BY-NC 4.0">
        </a>
        <br>
        <span style="font-size: 0.8em;">By Luís Simões da Cunha</span>
    </div>
    """, unsafe_allow_html=True)

#######################################
# 4) Create tabs for different sections
#######################################
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Interactive Tool", 
    "📚 Theory Behind the Model", 
    "📖 Comprehensive Tutorial", 
    "🛠️ Practical Labs",
    "🧠 The Very Basics of Options"
])

#######################################
# Tab 1: Interactive Tool
#######################################
with tab1:
    st.subheader("Interactive Binomial Pricing Calculator")
    
    # Compute option price and delta using the binomial model
    price, delta = binomial_option_pricing(S, K, T, r, sigma, steps, option_type)
    
    # Display results in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"### Option Price: €{price:,.2f}")
        st.markdown(f"**Delta:** `{delta:.3f}`")
    with col2:
        selected_variable = st.selectbox(
            "Select Variable to Visualize:",
            ["Option Price", "Delta"],
            index=0
        )
        # Plot variable against a range of underlying stock prices
        fig, ax = plt.subplots(figsize=(10, 5))
        S_range = np.linspace(50, 150, 100)
        values = []
        for s_val in S_range:
            val, d_val = binomial_option_pricing(s_val, K, T, r, sigma, steps, option_type)
            if selected_variable == "Option Price":
                values.append(val)
            else:
                values.append(d_val)
        ax.plot(S_range, values, color='darkorange', linewidth=2)
        ax.axvline(S, color='red', linestyle='--', label='Current Stock Price')
        ax.set_title(f"{selected_variable} vs. Stock Price", fontweight='bold')
        ax.set_xlabel("Stock Price (S)")
        ax.set_ylabel(selected_variable)
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

#######################################
# Tab 2: Theory Behind the Model
#######################################
with tab2:
    st.markdown("""
    ## Binomial Options Pricing Model: Theoretical Foundations
    
    The **Binomial Model** is a discrete-time method for pricing options by constructing a price tree (lattice). 
    
    **Key Concepts:**
    - **Discrete-Time Framework:** The model divides the time to expiration into a finite number of steps.
    - **Price Evolution:** At each step, the underlying asset price can move up by a factor *u* or down by a factor *d*.
    - **Risk-Neutral Pricing:** Under a risk-neutral measure, the expected return on the asset equals the risk-free rate, allowing us to discount expected payoffs.
    - **Delta Hedging:** The model naturally leads to the concept of delta—the sensitivity of the option price to small changes in the underlying asset—which is crucial for hedging.
    
    This model lays the groundwork for understanding continuous-time models like Black-Scholes.
    """)

#######################################
# Tab 3: Comprehensive Tutorial
#######################################
with tab3:
    st.markdown(r"""
## Comprehensive Tutorial on the Binomial Model

**Step 1: Setting Up the Tree**  
- Divide the time to expiration ($T$) into $N$ steps.  
- Compute the time interval: $\Delta t = \frac{T}{N}$.  
- Determine the up and down factors:  
  $u = e^{\sigma \sqrt{\Delta t}}$ and $d = \frac{1}{u}$.

**Step 2: Risk-Neutral Probabilities**  
- Calculate the risk-neutral probability $p$ as:  
  $p = \frac{e^{r \Delta t} - d}{u - d}$.

**Step 3: Option Valuation at Maturity**  
- At expiration, compute the option payoff at each node (e.g., for a call, $\max(S-K,0)$).

**Step 4: Backward Induction**  
- Discount the option values back to the present by recursively applying the risk-neutral expectation.

**Step 5: Delta Hedging**  
- Compute the hedge ratio (delta) at the initial node as:  
  $\Delta = \frac{V_{up} - V_{down}}{S \times (u - d)}$,  
  where $V_{up}$ and $V_{down}$ are the option values after one time step.

Experiment with different numbers of steps to see how the model converges toward continuous-time pricing.
""")

#######################################
# Tab 4: Practical Labs
#######################################
with tab4:
    st.header("🔬 Practical Option Labs")
    st.markdown("""
    In these labs, you will apply the Binomial Model to real-world scenarios:
    
    - **Lab 1: 2-Step Tree for a European Call**  
      Price a European call option using a basic 2-step tree and observe the delta hedging ratio.
    
    - **Lab 2: Convergence Analysis**  
      Increase the number of steps (e.g., 50 steps) to see how the binomial price converges toward the Black-Scholes price.
    
    - **Lab 3: Put Option Analysis**  
      Change the option type to a put and compare the pricing and delta to that of a call.
    """)
    
    lab_choice = st.radio(
        "Select a Lab to View:",
        ("Lab 1: 2-Step Tree", "Lab 2: Convergence Analysis", "Lab 3: Put Option Analysis"),
        index=0
    )
    
    if lab_choice == "Lab 1: 2-Step Tree":
        st.subheader("📊 Lab 1: 2-Step Tree for a European Call")
        st.markdown("""
        **Objective:**  
        Understand the basic 2-step binomial tree and how it produces an option price and delta.
        
        **Steps:**  
        1. Click **Set Lab 1 Parameters** to initialize the 2-step tree with typical parameters.
        2. Observe the computed option price and delta.
        3. Note how the tree structure helps determine the hedge ratio.
        """)
        st.button("⚡ Set Lab 1 Parameters", on_click=set_lab1_parameters, key="lab1_setup")
        
    elif lab_choice == "Lab 2: Convergence Analysis":
        st.subheader("📈 Lab 2: Convergence Analysis")
        st.markdown("""
        **Objective:**  
        Explore how increasing the number of steps refines the option price, demonstrating convergence toward continuous-time models.
        
        **Steps:**  
        1. Click **Set Lab 2 Parameters** to use a higher number of steps.
        2. Compare the option price and delta with those from a 2-step tree.
        3. Reflect on the benefits of finer time discretization.
        """)
        st.button("⚡ Set Lab 2 Parameters", on_click=set_lab2_parameters, key="lab2_setup")
        
    else:
        st.subheader("💹 Lab 3: Put Option Analysis")
        st.markdown("""
        **Objective:**  
        Analyze the pricing of a put option using the binomial tree and compare its delta with that of a call.
        
        **Steps:**  
        1. Click **Set Lab 3 Parameters** to switch the option type to put.
        2. Observe how the payoff and hedge ratio differ from the call option.
        3. Consider why put options behave differently under the same market conditions.
        """)
        st.button("⚡ Set Lab 3 Parameters", on_click=set_lab3_parameters, key="lab3_setup")

#######################################
# Tab 5: The Very Basics of Options
#######################################
with tab5:
    st.header("🧠 The Very Basics of Options")
    st.markdown("""
    **What Are Options?**  
    Options are contracts that give the holder the right, but not the obligation, to buy (call) or sell (put) an underlying asset at a specified price (strike) before a certain date (expiration).

    **Why Use the Binomial Model?**  
    - **Simplicity:** Provides a straightforward, discrete-time approach to option pricing.
    - **Intuition:** Helps understand risk-neutral pricing and delta hedging.
    - **Foundation:** Serves as a stepping stone to more advanced continuous-time models like Black-Scholes.

    **Key Takeaways:**  
    - The binomial model uses a tree structure to simulate potential future asset prices.
    - By moving backward through the tree, the model computes the present value of future payoffs.
    - Delta, the hedge ratio, quantifies how the option price changes with respect to changes in the underlying asset.
    """)

