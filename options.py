import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_delta(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type):
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    
    if option_type.lower() == 'call':
        delta = np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1)
    else:
        delta = -np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
    return delta

def black_scholes_gamma(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield):
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    
    gamma = np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) / \
            (spot_price * volatility * np.sqrt(time_to_expiry))
    return gamma

def black_scholes_vega(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield):
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    
    vega = spot_price * np.exp(-dividend_yield * time_to_expiry) * \
           norm.pdf(d1) * np.sqrt(time_to_expiry)
    return vega * 0.01

def black_scholes_theta(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type):
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    if option_type.lower() == 'call':
        theta = (-spot_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * volatility / 
                (2 * np.sqrt(time_to_expiry)) - 
                risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) +
                dividend_yield * spot_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1))
    else:
        theta = (-spot_price * np.exp(-dividend_yield * time_to_expiry) * norm.pdf(d1) * volatility / 
                (2 * np.sqrt(time_to_expiry)) + 
                risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) -
                dividend_yield * spot_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1))
    
    return theta / 365

def black_scholes_rho(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield, option_type):
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    if option_type.lower() == 'call':
        rho = strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:
        rho = -strike_price * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
    
    return rho * 0.01

def black_scholes_call_price(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield):
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    call_price = spot_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(d1) - \
                 strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    return call_price

def black_scholes_put_price(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, dividend_yield):
    d1 = (np.log(spot_price / strike_price) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / \
         (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    put_price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - \
                spot_price * np.exp(-dividend_yield * time_to_expiry) * norm.cdf(-d1)
    return put_price

def implied_volatility(option_price, spot_price, strike_price, time_to_expiry, risk_free_rate, dividend_yield, option_type='call'):
    if time_to_expiry <= 0 or option_price <= 0:
        return np.nan

    def objective_function(volatility):
        if option_type.lower() == 'call':
            theoretical_price = black_scholes_call_price(
                spot_price=spot_price,
                strike_price=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield
            )
        else:
            theoretical_price = black_scholes_put_price(
                spot_price=spot_price,
                strike_price=strike_price,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield
            )
        return theoretical_price - option_price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol