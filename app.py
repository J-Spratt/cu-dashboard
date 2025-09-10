import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

# Suppress DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- App Configuration ---
st.set_page_config(
    page_title="Credit Union Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .metric-card { background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; }
    h1, h2, h3, h4, h5 { color: #ffffff !important; }
    .css-1d391kg { background-color: #1e293b; }
    .recommendation { background-color: #1e293b; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .peer-metric { text-align: center; }
    .st-emotion-cache-16txtl3 { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- Core Calculation & Data Loading Functions ---
@st.cache_data
def load_and_process_data():
    """Loads, cleans, and calculates all necessary metrics from the source CSV."""
    def parse_currency(value):
        if pd.isna(value): return 0
        if isinstance(value, (int, float)): return value
        if isinstance(value, str):
            value = value.lower().replace('$', '').replace(',', '')
            multiplier = 1
            if 'b' in value: multiplier = 1e9; value = value.replace('b', '')
            elif 'mm' in value: multiplier = 1e6; value = value.replace('mm', '')
            elif 'm' in value: multiplier = 1e6; value = value.replace('m', '')
            elif 'k' in value: multiplier = 1e3; value = value.replace('k', '')
            try: return float(value) * multiplier
            except (ValueError, TypeError): return 0
        return 0

    try:
        df = pd.read_csv('cu_data.csv')
        currency_cols = [
            'Annual Interest Income', 'Assets Q1-2025', 'Assets Q1-2024', 'Loans Q1 - 2025',
            'Loans Q1 - 2024', 'Deposits Q1 - 2025', 'Deposits Q1-2024', 'Equity Capital Q1-2025',
            'Equity Capital Q1-2024', 'Loan Loss Allowance Q1-2025', 'Loan Loss Allowance Q1-2024',
            'Unbacked Non-Current Loans Q1 2025', 'Unbacked Non-Current Loans Q1 2024',
            'Real Estate Owned Q1 2025', 'Real Estate Owned Q1 2024', 'Assets per Employee'
        ]
        for col in currency_cols:
            if col in df.columns: df[col] = df[col].apply(parse_currency)

        other_numeric_cols = ['Employees', 'Year Chartered', 'NCUA #']
        for col in other_numeric_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        percentage_columns = [
            'Return on Assets - YTD', 'Return on Equity - YTD', 'Loan to Share Ratio',
            'Loan to Asset Ratio', 'Provision for Loan Loss Ratio', 'Loan Growth'
            ]
        for col in percentage_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce').fillna(0)

        df['Capital Adequacy Ratio'] = (df['Equity Capital Q1-2025'] / df['Assets Q1-2025'] * 100).fillna(0)
        df['Asset Growth'] = ((df['Assets Q1-2025'] - df['Assets Q1-2024']) / df['Assets Q1-2024'] * 100).fillna(0)
        df['Deposit Growth'] = ((df['Deposits Q1 - 2025'] - df['Deposits Q1-2024']) / df['Deposits Q1-2024'] * 100).fillna(0)
        df['Delinquency Ratio'] = ((df['Unbacked Non-Current Loans Q1 2025'] + df['Real Estate Owned Q1 2025']) / df['Assets Q1-2025'] * 100).fillna(0)
        df['Assets per Employee'] = (df['Assets Q1-2025'] / df['Employees']).fillna(0)

        df.replace([float('inf'), float('-inf')], 0, inplace=True)
        df['Health Score'] = df.apply(calculate_single_health_score, axis=1)

        return df
    except FileNotFoundError:
        st.error("Error: `cu_data.csv` not found."); return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}"); return pd.DataFrame()

def calculate_single_health_score(row):
    weights = {'capital': 0.35, 'growth': 0.25, 'roa': 0.25, 'delinquency': 0.15}
    cap_ratio = row.get('Capital Adequacy Ratio', 0)
    if cap_ratio >= 10: capital_score = 100
    elif cap_ratio >= 7: capital_score = 70 + (cap_ratio - 7) * 10
    else: capital_score = (cap_ratio / 7) * 70 if cap_ratio > 0 else 0
    growth = row.get('Asset Growth', 0)
    growth_score = max(0, min(100, 50 + (growth * 5)))
    roa = row.get('Return on Assets - YTD', 0)
    if roa >= 1: roa_score = 100
    elif roa >= 0.5: roa_score = 70 + (roa - 0.5) * 60
    else: roa_score = 70 + (roa * 70) if roa > -1 else 0
    delinquency = row.get('Delinquency Ratio', 0)
    delinquency_score = max(0, min(100, 100 - (delinquency * 50)))
    total_score = (capital_score * weights['capital'] + growth_score * weights['growth'] +
                   roa_score * weights['roa'] + delinquency_score * weights['delinquency'])
    return round(total_score)

def generate_recommendations(current, projected):
    recs = []
    if projected['Capital Adequacy Ratio'] < 7:
        recs.append("**Capital Adequacy Alert:** Projected ratio is below the well-capitalized threshold. Prioritize retaining earnings.")
    if projected['Delinquency Ratio'] > 1.5 and projected['Delinquency Ratio'] > current['Delinquency Ratio']:
        recs.append("**Delinquency Alert:** Projected increase is a concern. Review underwriting standards and collection efforts.")
    if projected['Assets per Employee'] < 4_000_000:
        recs.append("**Efficiency Alert:** Assets per employee are trending low. Look for opportunities to improve operational efficiency.")
    return recs if recs else ["**Stable Performance:** No critical recommendations triggered."]

# --- Main Application ---
st.title("Credit Union Performance Dashboard")
df = load_and_process_data()

if not df.empty:
    st.sidebar.header("Global Filters")
    search_name = st.sidebar.text_input("Search Credit Union Name")
    max_asset_val = df['Assets Q1-2025'].max()
    if pd.isna(max_asset_val) or max_asset_val == 0: max_asset_val = 10_000_000_000

    col1, col2 = st.sidebar.columns(2)
    min_assets = col1.number_input("Min Assets ($)", value=0, step=1_000_000, format="%d")
    max_assets = col2.number_input("Max Assets ($)", value=int(max_asset_val), step=1_000_000, format="%d")

    filtered_df = df[(df['Assets Q1-2025'] >= min_assets) & (df['Assets Q1-2025'] <= max_assets)]
    if search_name:
        filtered_df = filtered_df[filtered_df['Credit Union'].str.contains(search_name, case=False, na=False)]

    tab1, tab2, tab3 = st.tabs(["Dashboard", "Scenario Advisor", "Guide & Definitions"])

    with tab1:
        st.header("Overall Performance Snapshot")
        if not filtered_df.empty:
            st.write(f"Displaying **{len(filtered_df)}** of **{len(df)}** Credit Unions")
            st.dataframe(filtered_df.style.format(precision=2), use_container_width=True)
            
            st.markdown("---")
            st.header("Peer Analysis Tool")
            selected_cu_peer = st.selectbox("Select a Credit Union to Analyze", ["None"] + filtered_df['Credit Union'].unique().tolist(), key="main_peer_select")
            
            if selected_cu_peer != "None":
                cu_data = filtered_df[filtered_df['Credit Union'] == selected_cu_peer].iloc[0]
                peer_asset_range = st.slider(
                    "Select Peer Group Asset Size", 0, int(df['Assets Q1-2025'].max()),
                    (int(cu_data['Assets Q1-2025'] * 0.8), int(cu_data['Assets Q1-2025'] * 1.2)),
                    step=10_000_000, format="$%d", key="main_peer_slider"
                )
                peer_group = filtered_df[(filtered_df['Assets Q1-2025'].between(peer_asset_range[0], peer_asset_range[1])) & (filtered_df['Credit Union'] != selected_cu_peer)]

                if not peer_group.empty:
                    st.markdown(f"#### Comparison with Peer Group ({len(peer_group)} CUs)")
                    metrics_to_compare_list = [
                        'Health Score', 'Capital Adequacy Ratio', 'Asset Growth', 'Loan Growth', 'Deposit Growth',
                        'Return on Assets - YTD', 'Return on Equity - YTD', 'Delinquency Ratio', 
                        'Loan to Share Ratio', 'Loan to Asset Ratio', 'Provision for Loan Loss Ratio'
                    ]
                    peer_avg = peer_group[metrics_to_compare_list].mean()
                    
                    row1_cols = st.columns(6)
                    row2_cols = st.columns(5)
                    
                    def create_comparison_chart(metric_name, col, data, peer_data, fmt):
                         with col:
                            st.markdown(f"<h5 class='peer-metric'>{metric_name}</h5>", unsafe_allow_html=True)
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=[data], y=['You'], orientation='h', marker_color='#3b82f6', text=f"{data:{fmt}}", textposition='auto'))
                            fig.add_trace(go.Bar(x=[peer_data], y=['Peer Avg.'], orientation='h', marker_color='#4b5563', text=f"{peer_data:{fmt}}", textposition='auto'))
                            fig.update_layout(showlegend=False, height=120, margin=dict(l=5, r=5, t=20, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                            st.plotly_chart(fig, use_container_width=True)

                    create_comparison_chart("Health Score", row1_cols[0], cu_data['Health Score'], peer_avg['Health Score'], ".0f")
                    create_comparison_chart("Capital Adequacy", row1_cols[1], cu_data['Capital Adequacy Ratio'], peer_avg['Capital Adequacy Ratio'], ".2f")
                    create_comparison_chart("Asset Growth", row1_cols[2], cu_data['Asset Growth'], peer_avg['Asset Growth'], ".2f")
                    create_comparison_chart("Loan Growth", row1_cols[3], cu_data['Loan Growth'], peer_avg['Loan Growth'], ".2f")
                    create_comparison_chart("Deposit Growth", row1_cols[4], cu_data['Deposit Growth'], peer_avg['Deposit Growth'], ".2f")
                    create_comparison_chart("ROA", row1_cols[5], cu_data['Return on Assets - YTD'], peer_avg['Return on Assets - YTD'], ".2f")
                    
                    create_comparison_chart("ROE", row2_cols[0], cu_data['Return on Equity - YTD'], peer_avg['Return on Equity - YTD'], ".2f")
                    create_comparison_chart("Delinquency", row2_cols[1], cu_data['Delinquency Ratio'], peer_avg['Delinquency Ratio'], ".2f")
                    create_comparison_chart("Loan/Share", row2_cols[2], cu_data['Loan to Share Ratio'], peer_avg['Loan to Share Ratio'], ".2f")
                    create_comparison_chart("Loan/Asset", row2_cols[3], cu_data['Loan to Asset Ratio'], peer_avg['Loan to Asset Ratio'], ".2f")
                    create_comparison_chart("PLL Ratio", row2_cols[4], cu_data['Provision for Loan Loss Ratio'], peer_avg['Provision for Loan Loss Ratio'], ".2f")
                    
                    st.markdown("""
                        <div style="display: flex; justify-content: center; align-items: center; padding-top: 10px;">
                            <div style="display: flex; align-items: center; margin-right: 20px;">
                                <div style="width: 15px; height: 15px; background-color: #3b82f6; margin-right: 8px; border-radius: 3px;"></div>
                                <span>You</span>
                            </div>
                            <div style="display: flex; align-items: center;">
                                <div style="width: 15px; height: 15px; background-color: #4b5563; margin-right: 8px; border-radius: 3px;"></div>
                                <span>Peer Average</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.subheader("Metric Definitions")
                    def_col1, def_col2 = st.columns(2)
                    with def_col1:
                        st.markdown("**Health Score:** A proprietary score for a quick view of overall health. *(See the 'Guide & Definitions' tab for a full breakdown)*")
                        st.markdown("**Capital Adequacy:** Measures financial strength. Benchmark: >7%")
                        st.markdown("**Asset Growth:** Year-over-year % increase in assets. Benchmark: 3-8%")
                        st.markdown("**Loan Growth:** Year-over-year % increase in loans. Benchmark: 4-10%")
                        st.markdown("**Deposit Growth:** Year-over-year % increase in deposits. Benchmark: 3-8%")
                    with def_col2:
                        st.markdown("**ROA:** Profitability indicator. Benchmark: >1%")
                        st.markdown("**ROE:** Return for member-owners. Benchmark: >10%")
                        st.markdown("**Delinquency Ratio:** % of past-due loans. Benchmark: <1.25%")
                        st.markdown("**Loan/Share Ratio:** How effectively deposits are lent. Benchmark: >80%")
                        st.markdown("**Loan/Asset Ratio:** Proportion of assets that are loans. Benchmark: >65%")
                        st.markdown("**PLL Ratio:** Amount set aside for loan losses. Should be near Delinquency Ratio.")

                else:
                    st.warning("No other credit unions found in the selected asset range to form a peer group.")
        else:
            st.warning("No credit unions match the current filter criteria.")

    with tab2:
        st.header("Scenario Advisor")
        if not filtered_df.empty:
            cu_list_scenario = [""] + filtered_df['Credit Union'].tolist()
            selected_cu_scenario = st.selectbox("Select a CU for Scenario Planning", cu_list_scenario, key="advisor_select")
            if selected_cu_scenario:
                current_data = filtered_df[filtered_df['Credit Union'] == selected_cu_scenario].iloc[0]
                st.markdown(f"#### Projecting Q2 2025 for: **{selected_cu_scenario}**")
                
                proj_col1, proj_col2, proj_col3 = st.columns(3)

                with proj_col1:
                    proj_assets = st.number_input("Total Assets ($)", value=current_data['Assets Q1-2025'], format="%d")
                    proj_loans = st.number_input("Total Loans ($)", value=current_data['Loans Q1 - 2025'], format="%d")
                    proj_deposits = st.number_input("Total Deposits ($)", value=current_data['Deposits Q1 - 2025'], format="%d")
                with proj_col2:
                    proj_equity = st.number_input("Equity Capital ($)", value=current_data['Equity Capital Q1-2025'], format="%d")
                    proj_unbacked_loans = st.number_input("Unbacked Non-Current Loans ($)", value=current_data['Unbacked Non-Current Loans Q1 2025'], format="%d")
                    proj_reo = st.number_input("Real Estate Owned ($)", value=current_data['Real Estate Owned Q1 2025'], format="%d")
                with proj_col3:
                    proj_roa = st.number_input("ROA - YTD (%)", value=current_data['Return on Assets - YTD'], format="%.2f")
                    proj_employees = st.number_input("Number of Employees", value=int(current_data['Employees']))

                projected = {
                    'Capital Adequacy Ratio': (proj_equity / proj_assets * 100) if proj_assets else 0,
                    'Asset Growth': ((proj_assets - current_data['Assets Q1-2025']) / current_data['Assets Q1-2025'] * 100) if current_data['Assets Q1-2025'] else 0,
                    'Loan Growth': ((proj_loans - current_data['Loans Q1 - 2025']) / current_data['Loans Q1 - 2025'] * 100) if current_data['Loans Q1 - 2025'] else 0,
                    'Deposit Growth': ((proj_deposits - current_data['Deposits Q1 - 2025']) / current_data['Deposits Q1 - 2025'] * 100) if current_data['Deposits Q1 - 2025'] else 0,
                    'Delinquency Ratio': ((proj_unbacked_loans + proj_reo) / proj_assets * 100) if proj_assets else 0,
                    'Return on Assets - YTD': proj_roa,
                    'Assets per Employee': (proj_assets / proj_employees) if proj_employees else 0
                }
                projected_health_score = calculate_single_health_score(projected)

                st.markdown("---")
                st.markdown("#### Scenario Outcome & Recommendations")
                out_col1, out_col2 = st.columns([1, 2])
                out_col1.metric("Current Health Score (Q1)", f"{current_data['Health Score']:.0f}")
                out_col1.metric("Projected Health Score (Q2)", f"{projected_health_score:.0f}", delta=f"{projected_health_score - current_data['Health Score']:.0f}")
                recommendations = generate_recommendations(current_data, projected)
                for rec in recommendations:
                    out_col2.markdown(f"<div class='recommendation'>{rec}</div>", unsafe_allow_html=True)
        else:
            st.info("Filter for credit unions on the Main Dashboard to use the Scenario Advisor.")
    
    with tab3:
        st.header("Guide & Metric Definitions")
        st.markdown("""
        *Source: General standards are derived from National Credit Union Administration (NCUA) quarterly call report data and widely accepted industry best practices.*
        
        ---
        
        ### Core Health & Profitability
        
        **Health Score (0-100):** A proprietary score for a quick, holistic view of a credit union's overall health. It's a weighted average of:
        - **Capital Adequacy (35%):** Measures financial strength.
        - **Asset Growth (25%):** Reflects ability to attract new members/deposits.
        - **Return on Assets (25%):** A key profitability indicator.
        - **Delinquency (15%):** Measures loan portfolio quality.
        
        **Capital Adequacy Ratio (%):** Measures financial strength. A ratio of **7%** is considered "well-capitalized" by the NCUA.
        
        **Return on Assets - YTD (%):** A key profitability indicator. An ROA of **1% or higher** is generally considered excellent.
        
        **Return on Equity - YTD (%):** Measures the return generated for member-owners' equity. A **10%** ROE is a common target.
        
        ---
        ### Growth & Lending
        
        **Asset Growth (%):** The year-over-year percentage increase in total assets. **3% - 8%** is a healthy rate.
        
        **Loan Growth (%):** The year-over-year percentage increase in total loans. **4% - 10%** is a healthy and sustainable range.
        
        **Deposit Growth (%):** The year-over-year percentage increase in total deposits/shares. **3% - 8%** is a healthy rate.
        
        **Loan to Share Ratio (%):** Shows how effectively deposits are being lent out. **Above 80%** is typically strong.
        
        **Loan to Asset Ratio (%):** The proportion of assets that are loans. **>65%** indicates a focus on lending.
        
        ---
        ### Risk & Efficiency
        
        **Delinquency Ratio (%):** The percentage of the loan portfolio that is past due. **Below 1.25%** is generally healthy.
        
        **Provision for Loan Loss Ratio (%):** Amount set aside to cover potential loan losses. Should be close to the Delinquency Ratio.
        
        **Assets per Employee:** An efficiency metric. **$3.5M - $5M** per employee is a solid range.
        """)

else:
    st.warning("Dashboard could not be loaded. Please check the data source.")
