import asyncio
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from derebit import download_options_data
from options import (
    implied_volatility,
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_theta,
    black_scholes_rho
)

@st.cache_data(ttl=1800)
def fetch_options_data(selected_ticker, instruments):
    progress_bar = st.progress(0.0)
    progress_text = st.empty()
    
    def update_progress(completed, total):
        progress = float(completed) / float(total)
        progress_bar.progress(progress, text=f'Downloading options data: {completed}/{total} contracts')
        progress_text.text(f'Downloading options data: {completed}/{total} contracts')
        time.sleep(0.01)
    
    try:
        return asyncio.run(download_options_data(selected_ticker, instruments, max_concurrent=20, progress_callback=update_progress))
    finally:
        progress_bar.empty()
        progress_text.empty()

def create_volatility_surface(selected_ticker, instruments, option_type, min_strike_pct, max_strike_pct, y_axis_option, risk_free_rate, dividend_yield):
    try:
        with st.spinner('Fetching options data...'):
            options_df = fetch_options_data(selected_ticker, instruments)
        
        if options_df.empty:
            st.error('No option data available.')
            st.stop()
            
        options_df = options_df.copy()

        if option_type != 'Both':
            options_df = options_df[options_df['option_type'].str.lower() == option_type.lower()]
            if options_df.empty:
                st.error(f'No {option_type} options available.')
                st.stop()

        numeric_columns = ['strike', 'bid', 'ask', 'index_price']
        for col in numeric_columns:
            options_df.loc[:, col] = pd.to_numeric(options_df[col], errors='coerce')

        options_df = options_df.dropna()
            
        spot_price = float(options_df['index_price'].iloc[0])

        if selected_ticker in ['BTC', 'ETH']:
            options_df.loc[:, 'mid'] = (options_df['bid'] + options_df['ask']) / 2 * spot_price
        else:
            options_df.loc[:, 'mid'] = (options_df['bid'] + options_df['ask']) / 2

        options_df = options_df[
            (options_df['mid'] > 0) & 
            (options_df['bid'] > 0) & 
            (options_df['ask'] > 0)
        ]

        options_df = options_df[
            (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
            (options_df['strike'] <= spot_price * (max_strike_pct / 100))
        ]

        today = pd.Timestamp('today').normalize()
        options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'].astype(float) / 365.0

        options_df = options_df.copy()

        if options_df.empty:
            st.error('No options data available within the selected strike price range.')
            st.stop()

        options_df.reset_index(drop=True, inplace=True)

        with st.spinner('Calculating implied volatility...'):
            options_df['impliedVolatility'] = options_df.apply(
                lambda row: implied_volatility(
                    option_price=float(row['mid']),
                    spot_price=float(spot_price),
                    strike_price=float(row['strike']),
                    time_to_expiry=float(row['timeToExpiration']),
                    risk_free_rate=float(risk_free_rate),
                    dividend_yield=float(dividend_yield),
                    option_type=row['option_type'].lower()
                ), axis=1
            )

        options_df = options_df.dropna(subset=['impliedVolatility'])

        if options_df.empty:
            st.error('No valid implied volatility values calculated.')
            st.stop()

        options_df['impliedVolatility'] *= 100
        options_df = options_df.sort_values('strike')
        options_df['moneyness'] = options_df['strike'] / spot_price

        if len(options_df) < 3:
            st.error('Not enough data points to create a volatility surface.')
            st.stop()

        if y_axis_option == 'Strike Price ($)':
            Y = options_df['strike'].values
            y_label = 'Strike Price ($)'
        else:
            Y = options_df['moneyness'].values
            y_label = 'Moneyness (Strike / Spot)'

        X = options_df['timeToExpiration'].values
        Z = options_df['impliedVolatility'].values

        ti = np.linspace(X.min(), X.max(), 30)
        ki = np.linspace(Y.min(), Y.max(), 30)
        T, K = np.meshgrid(ti, ki)

        Zi = griddata((X, Y), Z, (T, K), method='linear')
        
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))

        T_days = T * 365 

        fig = go.Figure(data=[go.Surface(
            x=T_days,
            y=K,
            z=Zi,
            colorscale='Rainbow',
            colorbar_title='Implied Volatility (%)'
        )])

        fig.update_layout(
            title=f'Implied Volatility Surface for {selected_ticker} Options (Spot: ${spot_price:,.2f})',
            scene=dict(
                xaxis_title='Time to Expiration (days)', 
                yaxis_title=y_label,
                zaxis_title='Implied Volatility (%)'
            ),
            autosize=True,
            width=800,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        return fig

    except Exception as e:
        st.error(f'Error processing options data: {e}')
        return None 

def calculate_surface_greeks(selected_ticker, options_df, risk_free_rate, dividend_yield, min_strike_pct, max_strike_pct, option_type):
    options_df = options_df.copy()
    
    spot_price = float(options_df['index_price'].iloc[0])

    if selected_ticker in ['BTC', 'ETH']:
        options_df['mid'] = (options_df['bid'] + options_df['ask']) / 2 * spot_price
    else:
        options_df['mid'] = (options_df['bid'] + options_df['ask']) / 2

    options_df = options_df[
        (options_df['mid'] > 0) & 
        (options_df['bid'] > 0) & 
        (options_df['ask'] > 0)
    ]
    
    options_df = options_df[
        (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
        (options_df['strike'] <= spot_price * (max_strike_pct / 100))
    ]

    today = pd.Timestamp('today').normalize()
    options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
    options_df['timeToExpiration'] = options_df['daysToExpiration'].astype(float) / 365.0

    options_df['impliedVolatility'] = options_df.apply(
        lambda row: implied_volatility(
            option_price=float(row['mid']),
            spot_price=float(spot_price),
            strike_price=float(row['strike']),
            time_to_expiry=float(row['timeToExpiration']),
            risk_free_rate=float(risk_free_rate),
            dividend_yield=float(dividend_yield),
            option_type=row['option_type'].lower()
        ), axis=1
    )

    options_df['impliedVolatility'] *= 100

    options_df = options_df.dropna(subset=['impliedVolatility'])

    options_df = options_df[options_df['option_type'].str.lower() == option_type.lower()]
    
    expiry_groups = options_df.groupby('timeToExpiration')
    
    deltas = {}
    gammas = {}
    vegas = {}
    thetas = {}
    rhos = {}

    min_strike = options_df['strike'].min()
    max_strike = options_df['strike'].max()
    strikes = np.linspace(min_strike, max_strike, 100)
    
    for time_to_expiry, group in expiry_groups:
        group_spot_price = float(group['index_price'].iloc[0])

        delta_values = []
        gamma_values = []
        vega_values = []
        theta_values = []
        rho_values = []
        
        for strike in strikes:
            nearest_strike_row = group.iloc[(group['strike'] - strike).abs().argsort()[:1]]
            volatility = float(nearest_strike_row['impliedVolatility'].iloc[0]) / 100
            
            delta = black_scholes_delta(
                spot_price=group_spot_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                option_type=option_type
            )
            
            gamma = black_scholes_gamma(
                spot_price=group_spot_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield
            )
            
            vega = black_scholes_vega(
                spot_price=group_spot_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield
            )
            
            theta = black_scholes_theta(
                spot_price=group_spot_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                option_type=option_type
            )
            
            rho = black_scholes_rho(
                spot_price=group_spot_price,
                strike_price=strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
                option_type=option_type
            )
            
            delta_values.append(delta)
            gamma_values.append(gamma)
            vega_values.append(vega)
            theta_values.append(theta)
            rho_values.append(rho)
   
        expiry_label = f"T={time_to_expiry:.2f}y"
        deltas[expiry_label] = delta_values
        gammas[expiry_label] = gamma_values
        vegas[expiry_label] = vega_values
        thetas[expiry_label] = theta_values
        rhos[expiry_label] = rho_values
    
    return {
        'strikes': strikes,
        'moneyness': strikes/spot_price,
        'deltas': deltas,
        'gammas': gammas,
        'vegas': vegas,
        'thetas': thetas,
        'rhos': rhos
    }

def plot_greeks_surface(greeks_data, y_axis_option, option_type):
    tabs = st.tabs(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])

    colors = px.colors.qualitative.Set1
    
    x_values = greeks_data['moneyness'] if y_axis_option == 'Moneyness' else greeks_data['strikes']
    x_label = 'Moneyness (Strike/Spot)' if y_axis_option == 'Moneyness' else 'Strike Price ($)'
 
    def convert_expiry_label(label):
        years = float(label.split('=')[1].replace('y', ''))
        days = int(years * 365)
        return f"T={days}d"
    
    with tabs[0]:
        fig_delta = go.Figure()
        for i, (expiry, values) in enumerate(greeks_data['deltas'].items()):
            fig_delta.add_trace(go.Scatter(
                x=x_values,
                y=values,
                name=convert_expiry_label(expiry),  
                line=dict(color=colors[i % len(colors)])
            ))
        fig_delta.update_layout(
            title=f'Delta for {option_type} Options',
            xaxis_title=x_label,
            yaxis_title='Delta',
            showlegend=True
        )
        st.plotly_chart(fig_delta, use_container_width=True)
    
    with tabs[1]:
        fig_gamma = go.Figure()
        for i, (expiry, values) in enumerate(greeks_data['gammas'].items()):
            fig_gamma.add_trace(go.Scatter(
                x=x_values,
                y=values,
                name=convert_expiry_label(expiry), 
                line=dict(color=colors[i % len(colors)])
            ))
        fig_gamma.update_layout(
            title=f'Gamma for {option_type} Options',
            xaxis_title=x_label,
            yaxis_title='Gamma',
            showlegend=True
        )
        st.plotly_chart(fig_gamma, use_container_width=True)

    with tabs[2]:
        fig_vega = go.Figure()
        for i, (expiry, values) in enumerate(greeks_data['vegas'].items()):
            fig_vega.add_trace(go.Scatter(
                x=x_values,
                y=values,
                name=convert_expiry_label(expiry), 
                line=dict(color=colors[i % len(colors)])
            ))
        fig_vega.update_layout(
            title=f'Vega for {option_type} Options',
            xaxis_title=x_label,
            yaxis_title='Vega',
            showlegend=True
        )
        st.plotly_chart(fig_vega, use_container_width=True)
    
    with tabs[3]:
        fig_theta = go.Figure()
        for i, (expiry, values) in enumerate(greeks_data['thetas'].items()):
            fig_theta.add_trace(go.Scatter(
                x=x_values,
                y=values,
                name=convert_expiry_label(expiry),
                line=dict(color=colors[i % len(colors)])
            ))
        fig_theta.update_layout(
            title=f'Theta for {option_type} Options',
            xaxis_title=x_label,
            yaxis_title='Theta',
            showlegend=True
        )
        st.plotly_chart(fig_theta, use_container_width=True)
    
    with tabs[4]:
        fig_rho = go.Figure()
        for i, (expiry, values) in enumerate(greeks_data['rhos'].items()):
            fig_rho.add_trace(go.Scatter(
                x=x_values,
                y=values,
                name=convert_expiry_label(expiry), 
                line=dict(color=colors[i % len(colors)])
            ))
        fig_rho.update_layout(
            title=f'Rho for {option_type} Options',
            xaxis_title=x_label,
            yaxis_title='Rho',
            showlegend=True
        )
        st.plotly_chart(fig_rho, use_container_width=True)

def calculate_max_pain_by_expiry(options_df):
    max_pain_data = {}
    
    for expiry, group in options_df.groupby('expirationDate'):
        strikes = sorted(group['strike'].unique())
        total_pain = []
        call_pain = []
        put_pain = []
        
        for test_price in strikes:
            calls = group[group['option_type'] == 'call']
            call_pain_at_price = sum(
                calls['volume_usd'] * np.maximum(test_price - calls['strike'], 0)
            )
            
            puts = group[group['option_type'] == 'put']
            put_pain_at_price = sum(
                puts['volume_usd'] * np.maximum(puts['strike'] - test_price, 0)
            )
            
            total_pain.append(call_pain_at_price + put_pain_at_price)
            call_pain.append(call_pain_at_price)
            put_pain.append(put_pain_at_price)
        
        max_pain_data[expiry] = {
            'strikes': strikes,
            'total_pain': total_pain,
            'call_pain': call_pain,
            'put_pain': put_pain,
            'max_pain_strike': strikes[np.argmin(total_pain)],
            'call_volume': group[group['option_type'] == 'call'].groupby('strike')['volume_usd'].sum(),
            'put_volume': group[group['option_type'] == 'put'].groupby('strike')['volume_usd'].sum()
        }
    
    return max_pain_data

def plot_max_pain_analysis(max_pain_data, options_df):
    spot_price = float(options_df['index_price'].iloc[0])
    
    tabs = st.tabs([expiry.strftime('%Y-%m-%d') for expiry in max_pain_data.keys()])
    
    for tab, (expiry, data) in zip(tabs, max_pain_data.items()):
        with tab:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(x=data['strikes'],
                      y=data['call_volume'],
                      name='Calls Volume',
                      marker_color='rgb(0, 200, 0)',
                      opacity=0.7,
                      text=['C' for _ in data['strikes']],
                      textposition='inside'),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Bar(x=data['strikes'],
                      y=data['put_volume'],
                      name='Puts Volume',
                      marker_color='rgb(255, 50, 50)',
                      opacity=0.7,
                      text=['P' for _ in data['strikes']],
                      textposition='inside'),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=data['strikes'],
                          y=data['total_pain'],
                          name='Total Pain',
                          line=dict(color='blue', width=2)),
                secondary_y=True
            )
            
            fig.add_vline(x=data['max_pain_strike'],
                         line_dash="dash",
                         line_color='rgb(255, 140, 0)',
                         line_width=2)
            
            fig.add_vline(x=spot_price,
                         line_dash="dash",
                         line_color="purple",
                         line_width=2)
            
            fig.add_annotation(
                x=data['max_pain_strike'],
                y=1.05,
                yref="paper",
                text=f"Max Pain: {data['max_pain_strike']:,.0f}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color='rgb(255, 140, 0)')
            )
            
            fig.add_annotation(
                x=spot_price,
                y=1.12,
                yref="paper",
                text=f"Spot: {spot_price:,.0f}",
                showarrow=False,
                xanchor="left",
                xshift=10,
                font=dict(color='purple')
            )
            
            fig.update_layout(
                height=600,
                title=f'Max Pain Analysis for {expiry.strftime("%Y-%m-%d")}',
                showlegend=True,
                barmode='stack',
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(t=100)
            )
            
            fig.update_yaxes(title_text="Option Volume (USD)", secondary_y=False)
            fig.update_yaxes(title_text="Total Pain", secondary_y=True)
            fig.update_xaxes(title_text="Strike Price")
            
            fig.update_traces(
                hovertemplate="<br>".join([
                    "Strike: %{x}",
                    "Volume: %{y:,.0f}",
                    "Type: %{text}"
                ]),
                selector=dict(type='bar')
            )
            
            st.plotly_chart(fig, use_container_width=True)


