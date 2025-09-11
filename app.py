import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- App Configuration ---
st.set_page_config(
    page_title="Credit Union Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Constants ---
CURRENCY_COLUMNS = [
    'Annual Interest Income', 'Assets Q1-2025', 'Assets Q1-2024', 'Loans Q1 - 2025',
    'Loans Q1 - 2024', 'Deposits Q1 - 2025', 'Deposits Q1-2024', 'Equity Capital Q1-2025',
    'Equity Capital Q1-2024', 'Loan Loss Allowance Q1-2025', 'Loan Loss Allowance Q1-2024',
    'Unbacked Non-Current Loans Q1 2025', 'Unbacked Non-Current Loans Q1 2024',
    'Real Estate Owned Q1 2025', 'Real Estate Owned Q1 2024', 'Assets per Employee'
]

PERCENTAGE_COLUMNS = [
    'Return on Assets - YTD', 'Return on Equity - YTD', 'Loan to Share Ratio',
    'Loan to Asset Ratio', 'Provision for Loan Loss Ratio', 'Loan Growth'
]

NUMERIC_COLUMNS = ['Employees', 'Year Chartered', 'NCUA #']

HEALTH_SCORE_WEIGHTS = {
    'capital': 0.35,
    'growth': 0.25,
    'roa': 0.25,
    'delinquency': 0.15
}

# Benchmarks for recommendations
BENCHMARKS = {
    'capital_adequacy_min': 7,
    'capital_adequacy_optimal': 10,
    'delinquency_warning': 1.5,
    'delinquency_critical': 2.0,
    'assets_per_employee_min': 4_000_000,
    'roa_good': 0.5,
    'roa_excellent': 1.0,
    'loan_share_ratio_strong': 80,
    'asset_growth_min': 3,
    'asset_growth_max': 8
}

# --- Custom Styling ---
def apply_custom_styling():
    """Apply custom CSS styling to the dashboard"""
    st.markdown("""
    <style>
        .stApp { 
            background-color: #0f172a; 
        }
        .metric-card { 
            background-color: #1e293b; 
            padding: 20px; 
            border-radius: 10px; 
            border: 1px solid #334155; 
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #3b82f6;
        }
        h1, h2, h3, h4, h5 { 
            color: #ffffff !important; 
        }
        .css-1d391kg { 
            background-color: #1e293b; 
        }
        .recommendation { 
            background-color: #1e293b; 
            border-left: 5px solid #3b82f6; 
            padding: 15px; 
            border-radius: 5px; 
            margin-bottom: 10px; 
        }
        .recommendation-critical {
            border-left-color: #ef4444;
        }
        .recommendation-warning {
            border-left-color: #f59e0b;
        }
        .recommendation-info {
            border-left-color: #3b82f6;
        }
        .recommendation-success {
            border-left-color: #10b981;
        }
        .peer-metric { 
            text-align: center; 
        }
        .st-emotion-cache-16txtl3 { 
            padding-top: 2rem; 
        }
        /* Improve dataframe styling */
        .dataframe {
            font-size: 14px;
        }
        /* Add animation to metrics */
        .metric-value {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

# --- Utility Functions ---
def parse_currency(value) -> float:
    """Parse currency strings to float values with improved error handling"""
    if pd.isna(value):
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return 0.0
    
    # Clean the string
    value = value.lower().strip().replace('$', '').replace(',', '')
    
    # Handle different currency notations
    multipliers = {
        'b': 1e9,
        'bn': 1e9,
        'billion': 1e9,
        'mm': 1e6,
        'm': 1e6,
        'million': 1e6,
        'k': 1e3,
        'thousand': 1e3
    }
    
    for suffix, multiplier in multipliers.items():
        if suffix in value:
            value = value.replace(suffix, '').strip()
            try:
                return float(value) * multiplier
            except (ValueError, TypeError):
                return 0.0
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_process_data() -> pd.DataFrame:
    """Loads, cleans, and calculates all necessary metrics from the source CSV."""
    try:
        # Load data
        df = pd.read_csv('cu_data.csv')
        
        # Validate required columns exist
        missing_cols = []
        for col in ['Credit Union', 'Assets Q1-2025']:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return pd.DataFrame()
        
        # Process currency columns
        for col in CURRENCY_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(parse_currency)
        
        # Process numeric columns
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Process percentage columns
        for col in PERCENTAGE_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate derived metrics with safe division
        df['Capital Adequacy Ratio'] = np.where(
            df['Assets Q1-2025'] > 0,
            (df['Equity Capital Q1-2025'] / df['Assets Q1-2025']) * 100,
            0
        )
        
        df['Asset Growth'] = np.where(
            df['Assets Q1-2024'] > 0,
            ((df['Assets Q1-2025'] - df['Assets Q1-2024']) / df['Assets Q1-2024']) * 100,
            0
        )
        
        df['Deposit Growth'] = np.where(
            df['Deposits Q1-2024'] > 0,
            ((df['Deposits Q1 - 2025'] - df['Deposits Q1-2024']) / df['Deposits Q1-2024']) * 100,
            0
        )
        
        df['Delinquency Ratio'] = np.where(
            df['Assets Q1-2025'] > 0,
            ((df['Unbacked Non-Current Loans Q1 2025'] + df['Real Estate Owned Q1 2025']) / df['Assets Q1-2025']) * 100,
            0
        )
        
        df['Assets per Employee'] = np.where(
            df['Employees'] > 0,
            df['Assets Q1-2025'] / df['Employees'],
            0
        )
        
        # Handle infinities and NaN values
        df.replace([float('inf'), float('-inf')], 0, inplace=True)
        df.fillna(0, inplace=True)
        
        # Calculate health scores
        df['Health Score'] = df.apply(calculate_health_score, axis=1)
        
        # Add risk category
        df['Risk Category'] = df['Health Score'].apply(categorize_risk)
        
        return df
        
    except FileNotFoundError:
        st.error("⚠️ Error: `cu_data.csv` file not found. Please ensure the data file is in the correct location.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ An error occurred during data processing: {str(e)}")
        st.info("Please check that your CSV file is properly formatted.")
        return pd.DataFrame()

def calculate_health_score(row) -> int:
    """Calculate health score with improved scoring logic"""
    scores = {}
    
    # Capital Adequacy Score (0-100)
    cap_ratio = row.get('Capital Adequacy Ratio', 0)
    if cap_ratio >= BENCHMARKS['capital_adequacy_optimal']:
        scores['capital'] = 100
    elif cap_ratio >= BENCHMARKS['capital_adequacy_min']:
        scores['capital'] = 70 + ((cap_ratio - BENCHMARKS['capital_adequacy_min']) / 
                                  (BENCHMARKS['capital_adequacy_optimal'] - BENCHMARKS['capital_adequacy_min'])) * 30
    else:
        scores['capital'] = max(0, (cap_ratio / BENCHMARKS['capital_adequacy_min']) * 70)
    
    # Growth Score (0-100)
    growth = row.get('Asset Growth', 0)
    if BENCHMARKS['asset_growth_min'] <= growth <= BENCHMARKS['asset_growth_max']:
        scores['growth'] = 100
    elif growth < BENCHMARKS['asset_growth_min']:
        scores['growth'] = max(0, 50 + (growth * 10))
    else:
        scores['growth'] = max(0, 100 - ((growth - BENCHMARKS['asset_growth_max']) * 5))
    
    # ROA Score (0-100)
    roa = row.get('Return on Assets - YTD', 0)
    if roa >= BENCHMARKS['roa_excellent']:
        scores['roa'] = 100
    elif roa >= BENCHMARKS['roa_good']:
        scores['roa'] = 70 + ((roa - BENCHMARKS['roa_good']) / 
                              (BENCHMARKS['roa_excellent'] - BENCHMARKS['roa_good'])) * 30
    elif roa >= 0:
        scores['roa'] = (roa / BENCHMARKS['roa_good']) * 70
    else:
        scores['roa'] = max(0, 50 + (roa * 25))
    
    # Delinquency Score (0-100)
    delinquency = row.get('Delinquency Ratio', 0)
    if delinquency <= 0.5:
        scores['delinquency'] = 100
    elif delinquency <= BENCHMARKS['delinquency_warning']:
        scores['delinquency'] = 100 - ((delinquency - 0.5) / 
                                       (BENCHMARKS['delinquency_warning'] - 0.5)) * 30
    else:
        scores['delinquency'] = max(0, 70 - (delinquency - BENCHMARKS['delinquency_warning']) * 35)
    
    # Calculate weighted total
    total_score = sum(scores[key] * HEALTH_SCORE_WEIGHTS[key] for key in HEALTH_SCORE_WEIGHTS)
    
    return round(total_score)

def categorize_risk(health_score: float) -> str:
    """Categorize credit union based on health score"""
    if health_score >= 85:
        return "Excellent"
    elif health_score >= 70:
        return "Good"
    elif health_score >= 55:
        return "Fair"
    elif health_score >= 40:
        return "Needs Attention"
    else:
        return "Critical"

def generate_recommendations(current: pd.Series, projected: Dict) -> List[Tuple[str, str]]:
    """Generate recommendations with severity levels"""
    recs = []
    
    # Capital Adequacy Check
    if projected['Capital Adequacy Ratio'] < BENCHMARKS['capital_adequacy_min']:
        severity = "critical" if projected['Capital Adequacy Ratio'] < 5 else "warning"
        recs.append((
            f"**Capital Adequacy Alert:** Projected ratio ({projected['Capital Adequacy Ratio']:.1f}%) is below the well-capitalized threshold. "
            "Consider: 1) Retaining more earnings, 2) Limiting dividend payouts, 3) Raising additional capital.",
            severity
        ))
    
    # Delinquency Check
    if projected['Delinquency Ratio'] > BENCHMARKS['delinquency_warning']:
        if projected['Delinquency Ratio'] > current['Delinquency Ratio'] * 1.2:
            severity = "critical" if projected['Delinquency Ratio'] > BENCHMARKS['delinquency_critical'] else "warning"
            recs.append((
                f"**Delinquency Alert:** Projected delinquency ({projected['Delinquency Ratio']:.2f}%) shows concerning increase. "
                "Actions: 1) Review underwriting standards, 2) Enhance collection efforts, 3) Consider loan workout programs.",
                severity
            ))
    
    # Efficiency Check
    if projected['Assets per Employee'] < BENCHMARKS['assets_per_employee_min']:
        recs.append((
            f"**Efficiency Alert:** Assets per employee (${projected['Assets per Employee']:,.0f}) are below benchmark. "
            "Consider: 1) Process automation, 2) Digital transformation initiatives, 3) Strategic staffing review.",
            "warning"
        ))
    
    # Positive Indicators
    if projected['Return on Assets - YTD'] >= BENCHMARKS['roa_excellent']:
        recs.append((
            f"**Strong Performance:** Projected ROA ({projected['Return on Assets - YTD']:.2f}%) exceeds excellence benchmark. "
            "Maintain current strategies while exploring growth opportunities.",
            "success"
        ))
    
    if not recs:
        recs.append((
            "**Stable Performance:** All key metrics within acceptable ranges. Continue monitoring for optimization opportunities.",
            "info"
        ))
    
    return recs

def create_comparison_chart(metric_name: str, col, data: float, peer_data: float, fmt: str):
    """Create an enhanced comparison chart with better styling"""
    with col:
        st.markdown(f"<h5 class='peer-metric'>{metric_name}</h5>", unsafe_allow_html=True)
        
        # Determine color based on performance
        if "Delinquency" in metric_name:
            # Lower is better for delinquency
            color = '#10b981' if data < peer_data else '#ef4444'
        else:
            # Higher is better for most metrics
            color = '#10b981' if data > peer_data else '#f59e0b'
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[data], 
            y=['You'], 
            orientation='h', 
            marker_color=color, 
            text=f"{data:{fmt}}", 
            textposition='auto',
            hovertemplate=f'{metric_name}: {data:{fmt}}<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            x=[peer_data], 
            y=['Peer Avg.'], 
            orientation='h', 
            marker_color='#4b5563', 
            text=f"{peer_data:{fmt}}", 
            textposition='auto',
            hovertemplate=f'Peer Average: {peer_data:{fmt}}<extra></extra>'
        ))
        
        fig.update_layout(
            showlegend=False,
            height=120,
            margin=dict(l=5, r=5, t=20, b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_key_metrics(df: pd.DataFrame):
    """Display key metrics summary at the top of the dashboard"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_health = df['Health Score'].mean()
        st.metric("Avg Health Score", f"{avg_health:.0f}", 
                 delta=None, help="Average health score across all filtered CUs")
    
    with col2:
        total_assets = df['Assets Q1-2025'].sum()
        st.metric("Total Assets", f"${total_assets/1e9:.1f}B",
                 help="Combined assets of filtered CUs")
    
    with col3:
        avg_roa = df['Return on Assets - YTD'].mean()
        st.metric("Avg ROA", f"{avg_roa:.2f}%",
                 help="Average Return on Assets")
    
    with col4:
        avg_cap_ratio = df['Capital Adequacy Ratio'].mean()
        st.metric("Avg Capital Ratio", f"{avg_cap_ratio:.1f}%",
                 help="Average Capital Adequacy Ratio")
    
    with col5:
        at_risk = len(df[df['Risk Category'].isin(['Needs Attention', 'Critical'])])
        st.metric("CUs At Risk", at_risk,
                 help="Number of CUs needing attention or in critical condition")

# --- Main Application ---
def main():
    apply_custom_styling()
    
    st.title("📊 Credit Union Performance Dashboard")
    st.markdown("*Real-time analysis and scenario planning for credit union performance*")
    
    # Load data
    df = load_and_process_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Global Filters")
        
        # Search functionality
        search_name = st.text_input("Search Credit Union Name", 
                                   placeholder="Enter CU name...")
        
        # Asset range filter
        st.subheader("Asset Range")
        max_asset_val = df['Assets Q1-2025'].max()
        if pd.isna(max_asset_val) or max_asset_val == 0:
            max_asset_val = 10_000_000_000
        
        col1, col2 = st.columns(2)
        min_assets = col1.number_input("Min ($)", value=0, 
                                      step=1_000_000, format="%d")
        max_assets = col2.number_input("Max ($)", value=int(max_asset_val), 
                                      step=1_000_000, format="%d")
        
        # Risk category filter
        st.subheader("Risk Category")
        risk_categories = st.multiselect(
            "Select Categories",
            options=df['Risk Category'].unique().tolist(),
            default=df['Risk Category'].unique().tolist()
        )
        
        # Apply filters
        filtered_df = df[
            (df['Assets Q1-2025'] >= min_assets) & 
            (df['Assets Q1-2025'] <= max_assets) &
            (df['Risk Category'].isin(risk_categories))
        ]
        
        if search_name:
            filtered_df = filtered_df[
                filtered_df['Credit Union'].str.contains(search_name, case=False, na=False)
            ]
        
        st.markdown("---")
        st.info(f"📊 Showing {len(filtered_df)} of {len(df)} CUs")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔮 Scenario Advisor", 
                                      "📊 Analytics", "📚 Guide"])
    
    with tab1:
        if not filtered_df.empty:
            # Display key metrics
            st.header("Key Performance Indicators")
            display_key_metrics(filtered_df)
            
            st.markdown("---")
            
            # Display filtered data
            st.header("Credit Union Data")
            
            # Format columns for display
            display_df = filtered_df.copy()
            currency_format_cols = ['Assets Q1-2025', 'Loans Q1 - 2025', 
                                   'Deposits Q1 - 2025', 'Equity Capital Q1-2025']
            for col in currency_format_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
            
            percentage_format_cols = ['Capital Adequacy Ratio', 'Asset Growth', 
                                     'Return on Assets - YTD', 'Delinquency Ratio']
            for col in percentage_format_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
            
            # Display with conditional formatting
            st.dataframe(
                display_df[['Credit Union', 'Assets Q1-2025', 'Health Score', 
                          'Risk Category', 'Capital Adequacy Ratio', 
                          'Return on Assets - YTD', 'Asset Growth']],
                use_container_width=True,
                height=400
            )
            
            # Peer Analysis
            st.markdown("---")
            st.header("🎯 Peer Analysis Tool")
            
            selected_cu = st.selectbox(
                "Select a Credit Union to Analyze",
                ["None"] + filtered_df['Credit Union'].unique().tolist(),
                key="peer_select"
            )
            
            if selected_cu != "None":
                cu_data = filtered_df[filtered_df['Credit Union'] == selected_cu].iloc[0]
                
                # Dynamic peer group selection
                peer_selection_method = st.radio(
                    "Peer Group Selection Method",
                    ["Asset Size Range", "Top/Bottom Performers", "Custom Selection"],
                    horizontal=True
                )
                
                if peer_selection_method == "Asset Size Range":
                    peer_asset_range = st.slider(
                        "Select Peer Group Asset Range",
                        0, int(df['Assets Q1-2025'].max()),
                        (int(cu_data['Assets Q1-2025'] * 0.8), 
                         int(cu_data['Assets Q1-2025'] * 1.2)),
                        step=10_000_000,
                        format="$%d"
                    )
                    peer_group = filtered_df[
                        (filtered_df['Assets Q1-2025'].between(peer_asset_range[0], 
                                                               peer_asset_range[1])) &
                        (filtered_df['Credit Union'] != selected_cu)
                    ]
                
                elif peer_selection_method == "Top/Bottom Performers":
                    performance_metric = st.selectbox(
                        "Select Performance Metric",
                        ['Health Score', 'Return on Assets - YTD', 'Asset Growth']
                    )
                    top_n = st.slider("Number of Peers", 5, 20, 10)
                    peer_group = filtered_df[filtered_df['Credit Union'] != selected_cu].nlargest(
                        top_n, performance_metric
                    )
                
                else:  # Custom Selection
                    available_cus = filtered_df[filtered_df['Credit Union'] != selected_cu]['Credit Union'].tolist()
                    selected_peers = st.multiselect(
                        "Select Peer Credit Unions",
                        available_cus,
                        default=available_cus[:5] if len(available_cus) >= 5 else available_cus
                    )
                    peer_group = filtered_df[filtered_df['Credit Union'].isin(selected_peers)]
                
                if not peer_group.empty:
                    st.info(f"📊 Comparing with {len(peer_group)} peer credit unions")
                    
                    # Comparison metrics
                    metrics_to_compare = [
                        'Health Score', 'Capital Adequacy Ratio', 'Asset Growth',
                        'Loan Growth', 'Deposit Growth', 'Return on Assets - YTD',
                        'Return on Equity - YTD', 'Delinquency Ratio',
                        'Loan to Share Ratio', 'Loan to Asset Ratio',
                        'Provision for Loan Loss Ratio'
                    ]
                    
                    peer_avg = peer_group[metrics_to_compare].mean()
                    
                    # Display comparison charts
                    st.subheader("Performance Comparison")
                    
                    # First row of metrics
                    row1_cols = st.columns(6)
                    create_comparison_chart("Health Score", row1_cols[0], 
                                          cu_data['Health Score'], peer_avg['Health Score'], ".0f")
                    create_comparison_chart("Capital Adequacy", row1_cols[1],
                                          cu_data['Capital Adequacy Ratio'], 
                                          peer_avg['Capital Adequacy Ratio'], ".2f")
                    create_comparison_chart("Asset Growth", row1_cols[2],
                                          cu_data['Asset Growth'], peer_avg['Asset Growth'], ".2f")
                    create_comparison_chart("Loan Growth", row1_cols[3],
                                          cu_data['Loan Growth'], peer_avg['Loan Growth'], ".2f")
                    create_comparison_chart("Deposit Growth", row1_cols[4],
                                          cu_data['Deposit Growth'], peer_avg['Deposit Growth'], ".2f")
                    create_comparison_chart("ROA", row1_cols[5],
                                          cu_data['Return on Assets - YTD'],
                                          peer_avg['Return on Assets - YTD'], ".2f")
                    
                    # Second row of metrics
                    row2_cols = st.columns(5)
                    create_comparison_chart("ROE", row2_cols[0],
                                          cu_data['Return on Equity - YTD'],
                                          peer_avg['Return on Equity - YTD'], ".2f")
                    create_comparison_chart("Delinquency", row2_cols[1],
                                          cu_data['Delinquency Ratio'],
                                          peer_avg['Delinquency Ratio'], ".2f")
                    create_comparison_chart("Loan/Share", row2_cols[2],
                                          cu_data['Loan to Share Ratio'],
                                          peer_avg['Loan to Share Ratio'], ".2f")
                    create_comparison_chart("Loan/Asset", row2_cols[3],
                                          cu_data['Loan to Asset Ratio'],
                                          peer_avg['Loan to Asset Ratio'], ".2f")
                    create_comparison_chart("PLL Ratio", row2_cols[4],
                                          cu_data['Provision for Loan Loss Ratio'],
                                          peer_avg['Provision for Loan Loss Ratio'], ".2f")
                    
                    # Performance summary
                    st.markdown("---")
                    st.subheader("Performance Summary")
                    
                    outperforming = sum([
                        cu_data['Health Score'] > peer_avg['Health Score'],
                        cu_data['Capital Adequacy Ratio'] > peer_avg['Capital Adequacy Ratio'],
                        cu_data['Return on Assets - YTD'] > peer_avg['Return on Assets - YTD'],
                        cu_data['Asset Growth'] > peer_avg['Asset Growth'],
                        cu_data['Delinquency Ratio'] < peer_avg['Delinquency Ratio']
                    ])
                    
                    performance_level = "Outperforming" if outperforming >= 3 else "Underperforming"
                    performance_color = "🟢" if outperforming >= 3 else "🔴"
                    
                    st.markdown(f"### {performance_color} {selected_cu} is **{performance_level}** peers")
                    st.markdown(f"Outperforming in **{outperforming} of 5** key metrics")
                    
                else:
                    st.warning("No peer credit unions found with selected criteria.")
        else:
            st.warning("No credit unions match the current filter criteria.")
    
    with tab2:
        st.header("🔮 Scenario Planning & Advisor")
        
        if not filtered_df.empty:
            selected_cu_scenario = st.selectbox(
                "Select a Credit Union for Scenario Planning",
                [""] + filtered_df['Credit Union'].tolist(),
                key="scenario_select"
            )
            
            if selected_cu_scenario:
                current_data = filtered_df[filtered_df['Credit Union'] == selected_cu_scenario].iloc[0]
                
                st.markdown(f"### Projecting Q2 2025 for: **{selected_cu_scenario}**")
                st.info(f"Current Health Score: **{current_data['Health Score']:.0f}** ({current_data['Risk Category']})")
                
                # Scenario input columns
                st.subheader("📝 Adjust Projected Values")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Financial Position**")
                    proj_assets = st.number_input(
                        "Total Assets ($)",
                        value=float(current_data['Assets Q1-2025']),
                        format="%d",
                        help="Projected total assets for Q2"
                    )
                    proj_loans = st.number_input(
                        "Total Loans ($)",
                        value=float(current_data['Loans Q1 - 2025']),
                        format="%d",
                        help="Projected total loans outstanding"
                    )
                    proj_deposits = st.number_input(
                        "Total Deposits ($)",
                        value=float(current_data['Deposits Q1 - 2025']),
                        format="%d",
                        help="Projected total member deposits"
                    )
                
                with col2:
                    st.markdown("**Capital & Risk**")
                    proj_equity = st.number_input(
                        "Equity Capital ($)",
                        value=float(current_data['Equity Capital Q1-2025']),
                        format="%d",
                        help="Projected equity capital"
                    )
                    proj_unbacked_loans = st.number_input(
                        "Unbacked Non-Current Loans ($)",
                        value=float(current_data['Unbacked Non-Current Loans Q1 2025']),
                        format="%d",
                        help="Projected delinquent loans"
                    )
                    proj_reo = st.number_input(
                        "Real Estate Owned ($)",
                        value=float(current_data['Real Estate Owned Q1 2025']),
                        format="%d",
                        help="Projected REO assets"
                    )
                
                with col3:
                    st.markdown("**Performance & Efficiency**")
                    proj_roa = st.number_input(
                        "ROA - YTD (%)",
                        value=float(current_data['Return on Assets - YTD']),
                        format="%.2f",
                        help="Projected Return on Assets"
                    )
                    proj_employees = st.number_input(
                        "Number of Employees",
                        value=int(current_data['Employees']),
                        help="Projected employee count"
                    )
                    proj_loan_loss_provision = st.number_input(
                        "Loan Loss Provision Rate (%)",
                        value=float(current_data.get('Provision for Loan Loss Ratio', 0)),
                        format="%.2f",
                        help="Projected provision rate"
                    )
                
                # Calculate projected metrics
                projected = {
                    'Capital Adequacy Ratio': (proj_equity / proj_assets * 100) if proj_assets else 0,
                    'Asset Growth': ((proj_assets - current_data['Assets Q1-2025']) / 
                                   current_data['Assets Q1-2025'] * 100) if current_data['Assets Q1-2025'] else 0,
                    'Loan Growth': ((proj_loans - current_data['Loans Q1 - 2025']) / 
                                  current_data['Loans Q1 - 2025'] * 100) if current_data['Loans Q1 - 2025'] else 0,
                    'Deposit Growth': ((proj_deposits - current_data['Deposits Q1 - 2025']) / 
                                     current_data['Deposits Q1 - 2025'] * 100) if current_data['Deposits Q1 - 2025'] else 0,
                    'Delinquency Ratio': ((proj_unbacked_loans + proj_reo) / proj_assets * 100) if proj_assets else 0,
                    'Return on Assets - YTD': proj_roa,
                    'Assets per Employee': (proj_assets / proj_employees) if proj_employees else 0,
                    'Loan to Share Ratio': (proj_loans / proj_deposits * 100) if proj_deposits else 0,
                    'Provision for Loan Loss Ratio': proj_loan_loss_provision
                }
                
                # Calculate projected health score
                projected_health_score = calculate_health_score(pd.Series(projected))
                projected_risk_category = categorize_risk(projected_health_score)
                
                # Display results
                st.markdown("---")
                st.header("📊 Scenario Analysis Results")
                
                # Score comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Health Score",
                        f"{current_data['Health Score']:.0f}",
                        help=f"Risk Category: {current_data['Risk Category']}"
                    )
                
                with col2:
                    delta = projected_health_score - current_data['Health Score']
                    st.metric(
                        "Projected Health Score",
                        f"{projected_health_score:.0f}",
                        delta=f"{delta:+.0f}",
                        help=f"Risk Category: {projected_risk_category}"
                    )
                
                with col3:
                    if delta > 0:
                        impact = "Positive Impact ✅"
                        color = "green"
                    elif delta < 0:
                        impact = "Negative Impact ⚠️"
                        color = "red"
                    else:
                        impact = "Neutral Impact ➖"
                        color = "gray"
                    
                    st.markdown(f"<h3 style='color: {color};'>{impact}</h3>", unsafe_allow_html=True)
                
                # Key metrics comparison
                st.subheader("Key Metrics Comparison")
                
                metrics_comparison = pd.DataFrame({
                    'Metric': ['Capital Adequacy Ratio', 'Asset Growth', 'Delinquency Ratio', 
                              'ROA', 'Assets per Employee', 'Loan to Share Ratio'],
                    'Current': [
                        f"{current_data['Capital Adequacy Ratio']:.2f}%",
                        f"{current_data['Asset Growth']:.2f}%",
                        f"{current_data['Delinquency Ratio']:.2f}%",
                        f"{current_data['Return on Assets - YTD']:.2f}%",
                        f"${current_data['Assets per Employee']:,.0f}",
                        f"{current_data['Loan to Share Ratio']:.2f}%"
                    ],
                    'Projected': [
                        f"{projected['Capital Adequacy Ratio']:.2f}%",
                        f"{projected['Asset Growth']:.2f}%",
                        f"{projected['Delinquency Ratio']:.2f}%",
                        f"{projected['Return on Assets - YTD']:.2f}%",
                        f"${projected['Assets per Employee']:,.0f}",
                        f"{projected['Loan to Share Ratio']:.2f}%"
                    ]
                })
                
                st.dataframe(metrics_comparison, use_container_width=True, hide_index=True)
                
                # Recommendations
                st.markdown("---")
                st.subheader("Strategic Recommendations")
                
                recommendations = generate_recommendations(current_data, projected)
                
                for rec, severity in recommendations:
                    severity_class = f"recommendation-{severity}"
                    st.markdown(
                        f"<div class='recommendation {severity_class}'>{rec}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.info("Please select credit unions from the Dashboard tab to use the Scenario Advisor.")
    
    with tab3:
        st.header("📊 Advanced Analytics")
        
        if not filtered_df.empty:
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Distribution Analysis", "Correlation Analysis", "Trend Analysis", "Risk Analysis"]
            )
            
            if analysis_type == "Distribution Analysis":
                st.subheader("Distribution of Key Metrics")
                
                metric_to_analyze = st.selectbox(
                    "Select Metric",
                    ['Health Score', 'Capital Adequacy Ratio', 'Return on Assets - YTD',
                     'Asset Growth', 'Delinquency Ratio', 'Assets per Employee']
                )
                
                # Create distribution plot
                fig = px.histogram(
                    filtered_df,
                    x=metric_to_analyze,
                    nbins=20,
                    title=f"Distribution of {metric_to_analyze}",
                    labels={metric_to_analyze: metric_to_analyze},
                    color_discrete_sequence=['#3b82f6']
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{filtered_df[metric_to_analyze].mean():.2f}")
                with col2:
                    st.metric("Median", f"{filtered_df[metric_to_analyze].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{filtered_df[metric_to_analyze].std():.2f}")
                with col4:
                    st.metric("Range", f"{filtered_df[metric_to_analyze].max() - filtered_df[metric_to_analyze].min():.2f}")
            
            elif analysis_type == "Correlation Analysis":
                st.subheader("Correlation Matrix")
                
                corr_metrics = ['Health Score', 'Capital Adequacy Ratio', 'Asset Growth',
                              'Return on Assets - YTD', 'Delinquency Ratio', 'Loan Growth']
                
                corr_matrix = filtered_df[corr_metrics].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(x="Metrics", y="Metrics", color="Correlation"),
                    x=corr_metrics,
                    y=corr_metrics,
                    color_continuous_scale='RdBu',
                    aspect="auto",
                    title="Correlation Matrix of Key Metrics"
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key correlations
                st.subheader("Key Insights")
                strong_corr = []
                for i in range(len(corr_metrics)):
                    for j in range(i+1, len(corr_metrics)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_corr.append((corr_metrics[i], corr_metrics[j], corr_val))
                
                if strong_corr:
                    for metric1, metric2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                        direction = "positive" if corr > 0 else "negative"
                        st.write(f"• **{metric1}** and **{metric2}** show {direction} correlation ({corr:.2f})")
                else:
                    st.write("No strong correlations found (|r| > 0.5)")
            
            elif analysis_type == "Trend Analysis":
                st.subheader("Performance Trends")
                
                # Group by risk category
                risk_summary = filtered_df.groupby('Risk Category').agg({
                    'Health Score': 'mean',
                    'Capital Adequacy Ratio': 'mean',
                    'Return on Assets - YTD': 'mean',
                    'Asset Growth': 'mean',
                    'Credit Union': 'count'
                }).round(2)
                
                risk_summary.rename(columns={'Credit Union': 'Count'}, inplace=True)
                
                st.dataframe(risk_summary, use_container_width=True)
                
                # Visualize
                fig = px.bar(
                    risk_summary.reset_index(),
                    x='Risk Category',
                    y='Count',
                    title="Distribution by Risk Category",
                    color='Health Score',
                    color_continuous_scale='RdYlGn'
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Risk Analysis":
                st.subheader("Risk Assessment Matrix")
                
                # Create risk matrix
                fig = px.scatter(
                    filtered_df,
                    x='Capital Adequacy Ratio',
                    y='Delinquency Ratio',
                    size='Assets Q1-2025',
                    color='Health Score',
                    hover_data=['Credit Union'],
                    title="Risk Matrix: Capital vs Delinquency",
                    color_continuous_scale='RdYlGn'
                )
                
                # Add quadrant lines
                fig.add_hline(y=BENCHMARKS['delinquency_warning'], line_dash="dash", 
                            line_color="yellow", annotation_text="Warning Level")
                fig.add_vline(x=BENCHMARKS['capital_adequacy_min'], line_dash="dash",
                            line_color="yellow", annotation_text="Min Capital")
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk summary
                at_risk = filtered_df[
                    (filtered_df['Capital Adequacy Ratio'] < BENCHMARKS['capital_adequacy_min']) |
                    (filtered_df['Delinquency Ratio'] > BENCHMARKS['delinquency_warning'])
                ]
                
                if not at_risk.empty:
                    st.warning(f"⚠️ {len(at_risk)} credit unions are in high-risk zones")
                    st.dataframe(
                        at_risk[['Credit Union', 'Capital Adequacy Ratio', 
                               'Delinquency Ratio', 'Health Score']],
                        use_container_width=True
                    )
        else:
            st.info("Please select credit unions from the Dashboard tab to view analytics.")
    
    with tab4:
        st.header("📚 Guide & Metric Definitions")
        
        # Create expandable sections for better organization
        with st.expander("🎯 Quick Start Guide", expanded=True):
            st.markdown("""
            ### How to Use This Dashboard
            
            1. **Filter Data**: Use the sidebar to filter credit unions by name, asset size, or risk category
            2. **Analyze Performance**: Review key metrics and peer comparisons on the Dashboard tab
            3. **Plan Scenarios**: Use the Scenario Advisor to project future performance
            4. **Deep Dive**: Explore advanced analytics for detailed insights
            
            ### Key Features
            - **Real-time Health Scoring**: Proprietary algorithm evaluating multiple performance dimensions
            - **Peer Benchmarking**: Compare against similar institutions
            - **Scenario Planning**: Project future performance based on strategic changes
            - **Risk Assessment**: Identify and monitor at-risk institutions
            """)
        
        with st.expander("📊 Core Metrics", expanded=False):
            st.markdown(f"""
            ### Health & Capital Metrics
            
            **Health Score (0-100)**
            - Composite score measuring overall institutional health
            - Weights: Capital (35%), Growth (25%), ROA (25%), Delinquency (15%)
            - Scoring: 85+ Excellent | 70-84 Good | 55-69 Fair | 40-54 Needs Attention | <40 Critical
            
            **Capital Adequacy Ratio (%)**
            - Formula: (Equity Capital / Total Assets) × 100
            - Benchmark: {BENCHMARKS['capital_adequacy_min']}% minimum for "well-capitalized"
            - Target: {BENCHMARKS['capital_adequacy_optimal']}% or higher for optimal safety
            
            ### Profitability Metrics
            
            **Return on Assets - YTD (%)**
            - Measures profitability relative to total assets
            - Benchmark: {BENCHMARKS['roa_good']}% is good, {BENCHMARKS['roa_excellent']}% is excellent
            
            **Return on Equity - YTD (%)**
            - Return generated for member-owners' equity
            - Target: 10% or higher indicates strong performance
            """)
        
        with st.expander("📈 Growth & Efficiency Metrics", expanded=False):
            st.markdown(f"""
            ### Growth Indicators
            
            **Asset Growth (%)**
            - Year-over-year percentage change in total assets
            - Healthy range: {BENCHMARKS['asset_growth_min']}% - {BENCHMARKS['asset_growth_max']}%
            
            **Loan Growth (%)**
            - Year-over-year percentage change in loan portfolio
            - Sustainable range: 4% - 10%
            
            **Deposit Growth (%)**
            - Year-over-year percentage change in member deposits
            - Healthy range: 3% - 8%
            
            ### Efficiency Metrics
            
            **Assets per Employee**
            - Total Assets / Number of Employees
            - Benchmark: ${BENCHMARKS['assets_per_employee_min']:,.0f} minimum
            - Industry average: $3.5M - $5M per employee
            
            **Loan to Share Ratio (%)**
            - (Total Loans / Total Deposits) × 100
            - Strong performance: >{BENCHMARKS['loan_share_ratio_strong']}%
            """)
        
        with st.expander("⚠️ Risk Metrics", expanded=False):
            st.markdown(f"""
            ### Credit Risk Indicators
            
            **Delinquency Ratio (%)**
            - (Non-Current Loans + REO) / Total Assets × 100
            - Warning level: {BENCHMARKS['delinquency_warning']}%
            - Critical level: {BENCHMARKS['delinquency_critical']}%
            
            **Provision for Loan Loss Ratio (%)**
            - Reserves set aside for potential loan losses
            - Should align with delinquency levels
            - Higher provisions indicate conservative risk management
            
            ### Risk Categories
            
            Based on Health Score:
            - **Excellent (85-100)**: Strong performance across all metrics
            - **Good (70-84)**: Solid performance with minor areas for improvement
            - **Fair (55-69)**: Average performance, monitoring recommended
            - **Needs Attention (40-54)**: Below average, intervention suggested
            - **Critical (<40)**: Immediate attention required
            """)
        
        with st.expander("📖 Data Sources & Methodology", expanded=False):
            st.markdown("""
            ### Data Sources
            - Primary data from NCUA quarterly call reports
            - Financial metrics calculated from Q1 2025 and Q1 2024 data
            - Industry benchmarks based on peer group analysis
            
            ### Health Score Methodology
            
            The Health Score is calculated using a weighted average of four key components:
            
            1. **Capital Strength (35%)**: Based on capital adequacy ratio
            2. **Growth Performance (25%)**: Asset growth rate evaluation
            3. **Profitability (25%)**: Return on assets performance
            4. **Asset Quality (15%)**: Inverse of delinquency ratio
            
            Each component is scored 0-100 and combined using the specified weights.
            
            ### Important Notes
            - All financial figures are in USD
            - Percentages are annualized where applicable
            - Peer groups are determined by asset size similarity
            - Recommendations are algorithmic suggestions, not financial advice
            """)
        
        st.markdown("---")
        st.info("**Tip**: Use the scenario advisor to test strategic decisions before implementation")
        st.warning("**Disclaimer**: This dashboard provides analytical insights only. Always consult with financial professionals for decision-making.")

if __name__ == "__main__":
    main()
