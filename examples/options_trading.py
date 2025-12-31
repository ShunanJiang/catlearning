"""
Options trading strategies example.

Demonstrates:
1. Covered call strategy
2. Cash-secured put strategy
3. Options chain analysis
4. Greeks calculation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.data_manager import DataManager
from src.derivatives.options_strategies import (
    CoveredCallStrategy,
    CashSecuredPutStrategy,
    ProtectivePutStrategy
)


def main():
    """Run options trading example."""
    print("="*80)
    print("Options Trading Strategies Example")
    print("="*80)

    # Initialize data manager
    config = load_config("config.yaml")
    data_manager = DataManager(config.data_config)

    # Fetch stock data
    print("\nFetching QQQ market data...")
    stock_data = data_manager.get_market_data("QQQ", start_date="2023-01-01")
    current_price = stock_data.iloc[-1]['close']
    volatility = stock_data.iloc[-1].get('volatility_20', 0.20)

    print(f"Current QQQ price: ${current_price:.2f}")
    print(f"Current volatility: {volatility*100:.2f}%")

    # Fetch options chain
    print("\nFetching options chain...")
    try:
        options_chain = data_manager.get_options_data("QQQ")

        calls = options_chain['calls']
        puts = options_chain['puts']

        print(f"Found {len(calls)} call options and {len(puts)} put options")

        # Covered Call Strategy
        print("\n" + "="*80)
        print("COVERED CALL STRATEGY")
        print("="*80)

        cc_strategy = CoveredCallStrategy(config.strategies.get('options_income', {}))

        # Generate signals
        signals = cc_strategy.generate_signals(stock_data.tail(60))
        print(f"\nGenerated {(signals == 1).sum()} covered call signals")

        # Select optimal strike
        if not calls.empty:
            selected_call = cc_strategy.select_strike(
                calls,
                current_price,
                volatility
            )

            if selected_call is not None:
                print(f"\nRecommended Call Option:")
                print(f"  Strike: ${selected_call.get('strike', 0):.2f}")
                print(f"  Premium: ${selected_call.get('lastPrice', 0):.2f}")
                print(f"  Delta: {selected_call.get('delta', 0):.2f}")
                print(f"  Expiration: {selected_call.get('expiration', 'N/A')}")
                print(f"  DTE: {selected_call.get('dte', 0)} days")

        # Cash-Secured Put Strategy
        print("\n" + "="*80)
        print("CASH-SECURED PUT STRATEGY")
        print("="*80)

        csp_strategy = CashSecuredPutStrategy(config.strategies.get('options_income', {}))

        signals = csp_strategy.generate_signals(stock_data.tail(60))
        print(f"\nGenerated {(signals == 1).sum()} cash-secured put signals")

        if not puts.empty:
            selected_put = csp_strategy.select_strike(
                puts,
                current_price,
                volatility
            )

            if selected_put is not None:
                print(f"\nRecommended Put Option:")
                print(f"  Strike: ${selected_put.get('strike', 0):.2f}")
                print(f"  Premium: ${selected_put.get('lastPrice', 0):.2f}")
                print(f"  Delta: {selected_put.get('delta', 0):.2f}")
                print(f"  Expiration: {selected_put.get('expiration', 'N/A')}")
                print(f"  DTE: {selected_put.get('dte', 0)} days")

        # Protective Put Strategy
        print("\n" + "="*80)
        print("PROTECTIVE PUT STRATEGY")
        print("="*80)

        pp_strategy = ProtectivePutStrategy(config.strategies.get('options_income', {}))

        signals = pp_strategy.generate_signals(stock_data.tail(60))
        print(f"\nGenerated {(signals == 1).sum()} protective put signals")

        # Calculate Greeks for sample option
        print("\n" + "="*80)
        print("OPTION GREEKS EXAMPLE")
        print("="*80)

        strike = current_price * 1.05  # 5% OTM call
        tte = 30 / 365  # 30 days to expiration

        greeks = cc_strategy.calculate_option_greeks(
            spot=current_price,
            strike=strike,
            time_to_expiry=tte,
            volatility=volatility,
            option_type='call'
        )

        print(f"\nGreeks for ${strike:.2f} strike call (30 DTE):")
        print(f"  Delta: {greeks['delta']:.4f}")
        print(f"  Gamma: {greeks['gamma']:.4f}")
        print(f"  Theta: ${greeks['theta']:.2f} per day")
        print(f"  Vega: ${greeks['vega']:.2f} per 1% IV change")
        print(f"  Rho: ${greeks['rho']:.2f} per 1% rate change")

    except Exception as e:
        print(f"\nError fetching options data: {e}")
        print("Options data may not be available for this symbol")

    print("\n" + "="*80)
    print("Options analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
