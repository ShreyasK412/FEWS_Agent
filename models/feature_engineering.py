"""
Feature engineering for predictive risk assessment.
Extracts temporal, geographic, and composite indicators.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta


class FeatureEngineer:
    """Extract predictive features from food security data."""
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features: trends, volatility, seasonality.
        
        Features:
        - Price trends (3-month, 6-month moving averages)
        - Price volatility (standard deviation)
        - Seasonal patterns
        - Year-over-year comparisons
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["region", "date"])
        
        # Price trend features (if market_prices column exists)
        if "market_prices" in df.columns:
            # Extract average price from market_prices dict
            df["avg_price"] = df["market_prices"].apply(
                lambda x: np.mean(list(x.values())) if isinstance(x, dict) and x else np.nan
            )
            
            # Moving averages
            df["price_3m_ma"] = df.groupby("region")["avg_price"].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            df["price_6m_ma"] = df.groupby("region")["avg_price"].transform(
                lambda x: x.rolling(window=6, min_periods=1).mean()
            )
            
            # Volatility
            df["price_volatility"] = df.groupby("region")["avg_price"].transform(
                lambda x: x.rolling(window=6, min_periods=1).std()
            )
            
            # Year-over-year comparison
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["yoy_price_change"] = df.groupby(["region", "month"])["avg_price"].transform(
                lambda x: x.pct_change(periods=12)
            )
        
        # Seasonal features
        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
        df["lean_season"] = df["date"].dt.month.isin([6, 7, 8, 9]).astype(int)  # Jun-Sep
        
        return df
    
    def extract_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract geographic features: regional clustering, neighboring conditions.
        
        Features:
        - Regional clustering
        - Neighboring region conditions
        - Distance from conflict zones
        - Market accessibility
        """
        df = df.copy()
        
        # Regional risk clustering (average IPC phase in region)
        if "ipc_phase" in df.columns:
            df["regional_avg_ipc"] = df.groupby("region")["ipc_phase"].transform("mean")
            df["regional_max_ipc"] = df.groupby("region")["ipc_phase"].transform("max")
        
        # Neighboring region conditions (simplified - would need spatial join)
        # For now, use admin1 average
        if "admin_level" in df.columns:
            df["admin1_avg_risk"] = df.groupby("admin_level")["ipc_phase"].transform("mean")
        
        return df
    
    def extract_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract composite indicators.
        
        Indicators:
        - Food Consumption Score (FCS) - if available
        - Livelihood Coping Strategy Index (LCSI)
        - Market Functionality Index
        """
        df = df.copy()
        
        # Market Functionality Index (simplified)
        if "market_prices" in df.columns and "price_volatility" in df.columns:
            # Lower volatility = better market functionality
            df["market_functionality"] = 1 / (1 + df["price_volatility"].fillna(0))
        
        # Risk composite score
        risk_factors = []
        if "ipc_phase" in df.columns:
            risk_factors.append(df["ipc_phase"] / 5.0)  # Normalize to 0-1
        if "price_volatility" in df.columns:
            risk_factors.append(df["price_volatility"] / (df["price_volatility"].max() + 1e-6))
        if "conflict_incidents" in df.columns:
            risk_factors.append(df["conflict_incidents"] / (df["conflict_incidents"].max() + 1e-6))
        
        if risk_factors:
            df["composite_risk_score"] = np.mean(risk_factors, axis=0)
        
        return df
    
    def extract_leading_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract leading indicators for early warning.
        
        Indicators:
        - Rainfall deficits (2-3 months ahead)
        - Crop condition anomalies
        - Conflict escalation patterns
        - Displacement rate changes
        - Currency depreciation rates
        """
        df = df.copy()
        df = df.sort_values(["region", "date"])
        
        # Rainfall deficit (if available)
        if "rainfall_mm" in df.columns:
            df["rainfall_3m_avg"] = df.groupby("region")["rainfall_mm"].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            df["rainfall_12m_avg"] = df.groupby("region")["rainfall_mm"].transform(
                lambda x: x.rolling(window=12, min_periods=1).mean()
            )
            df["rainfall_deficit"] = (df["rainfall_3m_avg"] - df["rainfall_12m_avg"]) / (df["rainfall_12m_avg"] + 1e-6)
        
        # Conflict escalation (if available)
        if "conflict_incidents" in df.columns:
            df["conflict_3m_avg"] = df.groupby("region")["conflict_incidents"].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            df["conflict_12m_avg"] = df.groupby("region")["conflict_incidents"].transform(
                lambda x: x.rolling(window=12, min_periods=1).mean()
            )
            df["conflict_escalation"] = (df["conflict_3m_avg"] - df["conflict_12m_avg"]) / (df["conflict_12m_avg"] + 1e-6)
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering steps."""
        df = self.extract_temporal_features(df)
        df = self.extract_geographic_features(df)
        df = self.extract_composite_indicators(df)
        df = self.extract_leading_indicators(df)
        return df


if __name__ == "__main__":
    # Test
    sample_data = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "region": ["Tigray"] * 12,
        "ipc_phase": [2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 2, 2],
        "market_prices": [{"maize": 100 + i*10} for i in range(12)],
        "rainfall_mm": [50 + np.random.randn()*10 for _ in range(12)]
    })
    
    engineer = FeatureEngineer()
    features = engineer.engineer_all_features(sample_data)
    print(features.columns.tolist())

