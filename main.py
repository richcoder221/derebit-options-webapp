import asyncio
import streamlit as st
from data_processing import (
    create_volatility_surface,
    calculate_surface_greeks,
    plot_greeks_surface,
    fetch_options_data,
    calculate_max_pain_by_expiry,
    plot_max_pain_analysis
)
from derebit import get_all_options_list

st.set_page_config(
    page_title="Derebit Implied Volatility Surface and Greeks and Max Pain",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def fetch_instruments():
    return asyncio.run(get_all_options_list())

try:
    instruments = fetch_instruments()
except Exception as e:
    st.error(f"Error fetching instruments from Derebit: {e}")
    st.stop()

st.title('Derebit Implied Volatility Surface and Greeks and Max Pain')

st.sidebar.header('Ticker')
available_tickers = list(instruments.keys())
selected_ticker = st.sidebar.selectbox(
    'Select Ticker',
    available_tickers
)

if selected_ticker:
    available_dates = list(instruments[selected_ticker].keys())
    available_dates.sort()
    st.sidebar.write(f'Available Expiration Dates ({len(available_dates)}):')
    
    dates_container = st.sidebar.container()
    with dates_container:
        st.markdown(
            """
            <div style="height: 150px; overflow-y: scroll; background-color: #262730; padding: 10px; border-radius: 5px;">
            {}
            </div>
            """.format('<br>'.join(available_dates)),
            unsafe_allow_html=True
        )

st.sidebar.header('Chart Settings')

st.sidebar.markdown('##### Left Chart Display')
option_type1 = st.sidebar.selectbox(
    'Select Option Type (Left):',
    ('Call', 'Put', 'Both'),
    index=0,
    key='option_type_1'
)
curve_y_axis_option1 = st.sidebar.selectbox(
    'Select Y-axis for Volatility Surface (Left):',
    ('Strike Price ($)', 'Moneyness'),
    key='curve_y_axis_option1'
)
greek_x_axis_option1 = st.sidebar.selectbox(
    'Select X-axis for Greeks (Left):',
    ('Strike Price ($)', 'Moneyness'),
    key='greek_x_axis_option1'
)

st.sidebar.markdown('##### Right Chart Display')
option_type2 = st.sidebar.selectbox(
    'Select Option Type (Right):',
    ('Call', 'Put', 'Both'),
    index=1,
    key='option_type_2'
)
curve_y_axis_option2 = st.sidebar.selectbox(
    'Select Y-axis for Volatility Surface (Right):',
    ('Strike Price ($)', 'Moneyness'),
    key='curve_y_axis_option2'
)
greek_x_axis_option2 = st.sidebar.selectbox(
    'Select X-axis for Greeks (Right):',
    ('Strike Price ($)', 'Moneyness'),
    key='greek_x_axis_option2'
)

st.sidebar.header('Strike Price Filter Parameters')

min_strike_pct = st.sidebar.number_input(
    'Minimum Strike Price (% of Spot Price)',
    min_value=0.0,
    max_value=9999.0,
    value=70.0,
    step=1.0,
    format="%.1f"
)

max_strike_pct = st.sidebar.number_input(
    'Maximum Strike Price (% of Spot Price)',
    min_value=0.0,
    max_value=9999.0,
    value=150.0,
    step=1.0,
    format="%.1f"
)

if min_strike_pct >= max_strike_pct:
    st.sidebar.error('Minimum percentage must be less than maximum percentage.')
    st.stop()

st.sidebar.header('Model Parameters')
st.sidebar.write('Adjust the parameters for the Black-Scholes model.')

risk_free_rate = st.sidebar.number_input(
    'Risk-Free Rate (%)',
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    format="%.1f"
) / 100.0

dividend_yield = st.sidebar.number_input(
    'Dividend Yield (%)',
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    format="%.1f"
) / 100.0

generate_button = st.button('Display Volatility Surfaces and Greeks and Max Pain')

col1, col2 = st.columns(2)

if generate_button:
    with col1:
        st.header(f"{option_type1} Options Volatility Surface")
        fig1 = create_volatility_surface(
            selected_ticker=selected_ticker,
            option_type=option_type1,
            min_strike_pct=min_strike_pct,
            max_strike_pct=max_strike_pct,
            y_axis_option=curve_y_axis_option1,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            instruments=instruments
        )
        
        st.plotly_chart(fig1, use_container_width=True)

        st.header(f"{option_type1} Options Greeks")
        options_df = fetch_options_data(selected_ticker, instruments)
        greeks_data1 = calculate_surface_greeks(
            selected_ticker=selected_ticker,
            options_df=options_df,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type1.lower(),
            min_strike_pct=min_strike_pct,
            max_strike_pct=max_strike_pct
        )
        plot_greeks_surface(
            greeks_data1,
            y_axis_option=greek_x_axis_option1,
            option_type=option_type1
        )

    with col2:
        st.header(f"{option_type2} Options Volatility Surface")
        fig2 = create_volatility_surface(
            selected_ticker=selected_ticker,
            option_type=option_type2,
            min_strike_pct=min_strike_pct,
            max_strike_pct=max_strike_pct,
            y_axis_option=curve_y_axis_option2,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            instruments=instruments
        )
    
        st.plotly_chart(fig2, use_container_width=True)

        st.header(f"{option_type2} Options Greeks")
        greeks_data2 = calculate_surface_greeks(
            selected_ticker=selected_ticker,
            options_df=options_df,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type2.lower(),
            min_strike_pct=min_strike_pct,
            max_strike_pct=max_strike_pct
        )
        plot_greeks_surface(
            greeks_data2,
            y_axis_option=greek_x_axis_option1,
            option_type=option_type2
        )
      
    st.header("Max Pain Analysis")

    max_pain_data = calculate_max_pain_by_expiry(options_df)
    plot_max_pain_analysis(max_pain_data, options_df)