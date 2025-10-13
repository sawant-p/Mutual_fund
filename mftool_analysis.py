import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from mftool import Mftool # Absolute import

# Initialize Mftool globally for this module
try:
    mf = Mftool()
    # Map scheme codes to names for UI selection
    SCHEME_NAMES = {v: k for k, v in mf.get_scheme_codes().items()}
except Exception as e:
    st.error(f"Error initializing Mftool. Mutual Fund Analysis features may not work: {e}")
    mf = None
    SCHEME_NAMES = {}


def get_scheme_code_by_name(scheme_name):
    """Utility to get scheme code from a selected name."""
    if not mf:
        return None
    return {v: k for k, v in mf.get_scheme_codes().items()}.get(scheme_name)

def get_sentiment(text):
    """Performs sentiment analysis using TextBlob."""
    if not text:
        return "Neutral", 0.0
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

# --- Main Application Function for MFTool Module ---

def run_mftool_app():
    """Renders the Streamlit UI for the MFTool analysis features."""
    if not mf:
        st.warning("Cannot run MFTool analysis because the Mftool library failed to initialize.")
        return

    st.header("Mutual Fund Market Analysis (MFTool)")
    
    option = st.sidebar.selectbox(
        "Choose MFTool Feature",
        [
            "View Available Schemes", "Scheme Details", "Historical NAV", "Compare NAVs",
            "Average AUM", "Performance Heatmap", "Risk and Volatility Analysis",
            "Sentiment Analysis", "Fund Summary Generator"
        ],
        key="mftool_feature_select"
    )

    scheme_options = list(SCHEME_NAMES.keys())
    scheme_options.insert(0, "Select a Scheme")
    
    # --- Feature Implementations (Restored from your original app.py) ---

    if option == 'View Available Schemes':
        st.subheader('View Available Schemes')
        amc = st.text_input("Enter AMC Name (e.g., HDFC, ICICI)", "ICICI")
        schemes = mf.get_available_schemes(amc)
        if schemes:
            st.dataframe(pd.DataFrame(schemes.items(), columns=["Scheme Code", "Scheme Name"]))
        else:
            st.write("No schemes found for the given AMC.")

    elif option == 'Scheme Details':
        st.subheader("Scheme Details")
        selected_scheme = st.selectbox("Select Scheme Name", options=scheme_options, key="detail_scheme_select")
        if selected_scheme != "Select a Scheme":
            scheme_code = get_scheme_code_by_name(selected_scheme)
            details = mf.get_scheme_details(scheme_code)
            if details:
                st.json(details)
            else:
                st.write("Could not fetch details.")

    elif option == 'Historical NAV':
        st.subheader("Historical NAV Chart")
        selected_scheme = st.selectbox("Select Scheme Name", options=scheme_options, key="nav_scheme_select")
        if selected_scheme != "Select a Scheme":
            scheme_code = get_scheme_code_by_name(selected_scheme)
            try:
                nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                if not nav_data.empty:
                    nav_data.reset_index(inplace=True)
                    # FIX: Explicitly select the first two columns to avoid the "Length mismatch" error
                    nav_data = nav_data.iloc[:, :2] 
                    nav_data.columns = ["date", "nav"] 
                    nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
                    nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
                    nav_data.set_index("date", inplace=True)
                    fig = px.line(nav_data, y="nav", title=f"Historical NAV Trend for {selected_scheme}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No NAV data to display for the selected scheme.")
            except Exception as e:
                st.write(f"Error fetching NAV data: {e}")


    elif option == 'Compare NAVs':
        st.header('Compare NAVs')
        selected_schemes = st.multiselect("Select schemes to compare", options=list(SCHEME_NAMES.keys()))
        if selected_schemes:
            comparison_df = pd.DataFrame()
            for scheme in selected_schemes:
                code = get_scheme_code_by_name(scheme)
                data = mf.get_scheme_historical_nav(code, as_Dataframe=True)
                if data is not None and not data.empty:
                    data = data.reset_index().rename(columns={"index": "date"})
                    # FIX: Explicitly select the first two columns to avoid the "Length mismatch" error
                    data = data.iloc[:, :2] 
                    data.columns = ['date', 'nav']
                    data["date"] = pd.to_datetime(data["date"], dayfirst=True).sort_values()
                    data["nav"] = pd.to_numeric(data["nav"], errors='coerce').replace(0, np.nan).interpolate()
                    comparison_df[scheme] = data.set_index("date")["nav"]
            
            if not comparison_df.empty:
                comparison_df = comparison_df.apply(lambda x: 100 * x / x.iloc[0], axis=0)
                fig = px.line(comparison_df, title="Normalized Comparison of NAVs")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid historical data found for selected schemes.")
        else:
            st.info("Select at least one scheme")

    elif option == 'Average AUM':
        st.header('Average AUM')
        # Use a relevant quarter for the request
        quarter = st.selectbox("Select Quarter", ["July - September 2024", "April - June 2024", "January - March 2024"])
        aum_data = mf.get_average_aum(quarter, False)
        if aum_data:
            aum_df = pd.DataFrame(aum_data)
            aum_df["Total AUM"] = aum_df[["AAUM Overseas", "AAUM Domestic"]].astype(float).sum(axis=1)
            st.dataframe(aum_df[["Fund Name", "Total AUM"]])
        else:
            st.write(f"No AUM data available for {quarter}")

    elif option == 'Performance Heatmap':
        st.header("Performance Heatmap")
        selected_scheme = st.selectbox("Select a Scheme", scheme_options, key="heatmap_scheme_select")
        if selected_scheme != "Select a Scheme":
            scheme_code = get_scheme_code_by_name(selected_scheme)
            try:
                nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                if not nav_data.empty:
                    nav_data = nav_data.reset_index().rename(columns={"index": "date"})
                    nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
                    nav_data["month"] = nav_data['date'].dt.month
                    nav_data['nav'] = pd.to_numeric(nav_data['nav'], errors='coerce')
                    nav_data = nav_data.sort_values('date').dropna(subset=['nav'])
                    nav_data['dayChange'] = nav_data['nav'].pct_change() * 100
                    
                    heatmap_data = nav_data.groupby("month")["dayChange"].mean().reset_index()
                    heatmap_data["month"] = heatmap_data["month"].astype(str)
                    
                    fig = px.density_heatmap(heatmap_data, x="month", y="dayChange", 
                                             title="NAV Performance Heatmap (Avg. Monthly Daily Change)", 
                                             color_continuous_scale="viridis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No historical data available")
            except Exception as e:
                st.write(f"Error generating heatmap: {e}")

    elif option == "Risk and Volatility Analysis":
        st.header("Risk and Volatility Analysis (Monte Carlo Simulation)")
        scheme_name = st.selectbox("Select a Scheme", scheme_options, key="risk_scheme_select")
        if scheme_name != "Select a Scheme":
            scheme_code = get_scheme_code_by_name(scheme_name)
            try:
                nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
                if not nav_data.empty:
                    nav_data = nav_data.reset_index().rename(columns={"index": "date"})
                    nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
                    nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce").dropna()
                    nav_data["returns"] = nav_data["nav"] / nav_data["nav"].shift(1) - 1 
                    nav_data = nav_data.dropna(subset=["returns"])
                    
                    # Calculate Metrics
                    ann_volatility = nav_data["returns"].std() * np.sqrt(252)
                    ann_return = (1 + nav_data["returns"].mean()) ** 252 - 1
                    risk_free_rate = 0.06 # Assumed constant
                    sharpe = (ann_return - risk_free_rate) / ann_volatility
                    
                    st.subheader(f"Metrics for {scheme_name}")
                    col_ret, col_vol, col_sharp = st.columns(3)
                    col_ret.metric("Annualized Return", f"{ann_return:.2%}")
                    col_vol.metric("Annualized Volatility", f"{ann_volatility:.2%}")
                    col_sharp.metric("Sharpe Ratio", f"{sharpe:.2f}")

                    # Monte Carlo Simulation
                    st.markdown("### Monte Carlo Simulation for Future NAV Projection")
                    col_sim, col_day = st.columns(2)
                    num_sim = col_sim.slider("Number of Simulations", 100, 5000, 1000)
                    num_days = col_day.slider("Projection Period (Days)", 30, 365, 252)
                    
                    last_nav = nav_data["nav"].iloc[-1]
                    daily_vol = nav_data["returns"].std()
                    daily_ret = nav_data["returns"].mean()
                    
                    simulations = []
                    for _ in range(num_sim):
                        prices = [last_nav]
                        for _ in range(num_days):
                            ret = np.random.normal(daily_ret, daily_vol)
                            prices.append(prices[-1] * (1 + ret))
                        simulations.append(prices)
                        
                    sim_df = pd.DataFrame(simulations).T
                    sim_df.index.name = "Day"
                    sim_df.columns = [f"Simulation {i+1}" for i in range(num_sim)]
                    st.line_chart(sim_df)
                else:
                    st.write("No historical data available")
            except Exception as e:
                st.write(f"Error generating risk and volatility analysis: {e}")

    elif option == "Sentiment Analysis":
        st.header("📰 Mutual Fund News Sentiment")
        fund_house = st.text_input("Enter Fund House (e.g., HDFC, ICICI)", "ICICI")

        if fund_house:
            try:
                rss_url = f"https://news.google.com/rss/search?q={fund_house}+mutual+fund&hl=en-IN&gl=IN&ceid=IN:en"
                response = requests.get(rss_url)
                soup = BeautifulSoup(response.content, features="xml")
                items = soup.findAll('item')

                headlines = [item.title.text for item in items]

                if headlines:
                    st.subheader(f"Fetched Headlines for {fund_house}")
                    st.dataframe(pd.DataFrame({"Headline": headlines}))

                    sentiments = []
                    for headline in headlines:
                        blob = TextBlob(headline)
                        polarity = blob.sentiment.polarity
                        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
                        sentiments.append({"Headline": headline, "Polarity": polarity, "Sentiment": sentiment})

                    df = pd.DataFrame(sentiments)
                    st.subheader("Sentiment Analysis Results")
                    st.dataframe(df)
                    fig = px.pie(df, names="Sentiment", title="Sentiment Distribution of News Headlines")
                    st.plotly_chart(fig)
                else:
                    st.warning("No headlines found. Try a different keyword.")
            except Exception as e:
                st.error(f"Error fetching news or performing sentiment analysis: {e}")


    elif option == "Fund Summary Generator":
        st.header("🧾 Fund Summary Generator")
        selected_scheme = st.selectbox("Select a Scheme", scheme_options, key="summary_scheme_select")
        if selected_scheme != "Select a Scheme":
            scheme_code = get_scheme_code_by_name(selected_scheme)
            try:
                nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

                if nav_data is None or nav_data.empty:
                    st.warning("No historical NAV data available for this scheme.")
                    return
                
                nav_data = nav_data.reset_index().rename(columns={"index": "date"})
                nav_data["date"] = pd.to_datetime(nav_data["date"], dayfirst=True)
                nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
                nav_data = nav_data.dropna().sort_values('date')

                returns = nav_data["nav"].pct_change().dropna()
                annual_return = (1 + returns.mean()) ** 252 - 1
                volatility = returns.std() * np.sqrt(252)

                risk_type = "Low" if volatility < 0.10 else "Moderate" if volatility < 0.20 else "High"
                return_type = "High" if annual_return > 0.12 else "Moderate" if annual_return > 0.08 else "Low"

                summary = f"""
                ### Summary for {selected_scheme}
                - **Average Annual Return**: `{annual_return:.2%}`  
                - **Annual Volatility (Risk)**: `{volatility:.2%}`  
                - **Interpreted Risk Level**: **{risk_type}** - **Interpreted Return Potential**: **{return_type}** *Overall*: This fund has shown **{return_type.lower()}** returns with a **{risk_type.lower()}** risk profile historically.
                """
                st.markdown(summary)

            except Exception as e:
                st.error(f"Could not generate summary: {e}")