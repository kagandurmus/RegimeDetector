import os
import json
import logging
from datetime import datetime

# ==========================================
# IMPORT DECOUPLED MODULES
# ==========================================
from src.ingestion import DataIngestor
from src.features import FeatureFactory
from src.engine import RegimeInterpreter
from src.sentiment import SentimentEngine
from src.narrative import NarrativeEngine  # Replacing 'main.py'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_full_pipeline():
    """Executes the decoupled Quant Pipeline from Ingestion to LLM Narrative."""
    logging.info("🚀 STARTING FULL QUANT PIPELINE")
    start_time = datetime.now()

    try:
        # ==========================================
        # PHASE 1: DATA INGESTION
        # ==========================================
        logging.info("--- Phase 1: Ingesting Live Market Data ---")
        os.makedirs("data/raw", exist_ok=True)
        
        ingestor = DataIngestor(ticker="SPY")
        raw_df = ingestor.get_unified_snapshot()
        
        if raw_df is None or raw_df.is_empty():
            raise ValueError("Ingestion failed: Retrieved empty DataFrame.")
            
        raw_df.write_parquet("data/raw/market_snapshot.parquet")
        logging.info("✅ Phase 1 Complete: Raw data saved.")

        # ==========================================
        # PHASE 2: FEATURE ENGINEERING & HMM
        # ==========================================
        logging.info("--- Phase 2: Engineering Indicators & Regimes ---")
        os.makedirs("data/processed", exist_ok=True)
        
        factory = FeatureFactory(input_path="data/raw/market_snapshot.parquet")
        factory.generate_indicators()
        processed_df = factory.fit_regimes()
        
        processed_df.write_parquet("data/processed/regime_data.parquet")
        logging.info("✅ Phase 2 Complete: Features and HMM labels saved.")

        # ==========================================
        # PHASE 3: PREDICTIVE FORECASTER (XGBoost)
        # ==========================================
        logging.info("--- Phase 3: Forecaster Inference ---")
        interpreter = RegimeInterpreter(data_path="data/processed/regime_data.parquet")
        
        # Unpack the inference outputs directly from the engine
        feats, shap_vals, pred_regime, conf, base_val = interpreter.train_interpreter() 
        logging.info(f"✅ Phase 3 Complete: T+1 Regime Forecast -> {pred_regime}")

        # Extract contextual metrics from the exact same processed dataframe
        latest_row = processed_df.tail(1).to_dicts()[0]
        today_rsi = float(latest_row.get("RSI", 50.0))
        
        if today_rsi < 30:
            rsi_condition = "OVERSOLD"
        elif today_rsi > 70:
            rsi_condition = "OVERBOUGHT"
        else:
            rsi_condition = "NEUTRAL"

        # Build the structured Math Dictionary for the LLM
        quant_data = {
            "predicted_regime": pred_regime,
            "confidence": round(float(conf), 4),
            "today_rsi": round(today_rsi, 2),
            "rsi_condition": rsi_condition,
            "shap_drivers": dict(zip(feats, [round(float(v), 4) for v in shap_vals])),
            "vol_5d": round(float(latest_row.get("Vol_5d", 0)), 4),
            "momentum_10d": round(float(latest_row.get("Momentum_10d", 0)), 4)
        }

        # ==========================================
        # PHASE 4: SENTIMENT GATHERING
        # ==========================================
        logging.info("--- Phase 4: Fetching Market Sentiment ---")
        sentiment_engine = SentimentEngine()
        sentiment_data = sentiment_engine.get_full_report()
        logging.info("✅ Phase 4 Complete: Live catalysts fetched.")

        # ==========================================
        # PHASE 5: AI NARRATIVE & SYNTHESIS
        # ==========================================
        logging.info("--- Phase 5: Generating AI Strategic Briefing ---")
        narrative_engine = NarrativeEngine(model_name="qwen3:14b")
        
        # Inject the decoupled data into the LLM logic
        final_narrative = narrative_engine.generate_briefing(quant_data, sentiment_data)

        # ==========================================
        # ADDITIONS FOR DASHBOARD COMPATIBILITY
        # ==========================================
        # 1. Create the Confidence Map for the Dashboard Bar Chart using returned 'conf'
        # This maps the winning confidence to the predicted regime and splits the rest.
        probs = [0.0, 0.0, 0.0]
        probs[int(pred_regime)] = float(conf)
        remaining = (1.0 - float(conf)) / 2
        for i in range(3):
            if i != int(pred_regime): probs[i] = remaining
            
        conf_map = {
            "Bear_Prob": probs[0],
            "Neutral_Prob": probs[1],
            "Bull_Prob": probs[2]
        }

        # 2. Extract the specific names dashboard.py looks for
        volatility_val = float(latest_row.get("Volatility", 0.0))
        vpa_val = float(latest_row.get("VPA_Pressure", 0.0))

        # ==========================================
        # FINAL: SAVE STATE FOR DASHBOARD / TELEGRAM
        # ==========================================
        dashboard_state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": {
                "regime": int(pred_regime),
                "confidence": conf_map # Changed to dict to satisfy Dashboard chart
            },
            "indicators": {
                "rsi": quant_data["today_rsi"],
                "volatility": volatility_val, # Required by dashboard.py line 71
                "vpa": vpa_val,               # Required by dashboard.py line 72
                "fear_greed": sentiment_data.get("fear_greed", 50)
            },
            "shap": quant_data["shap_drivers"],
            "narrative": final_narrative,
            "catalysts": sentiment_data.get("catalysts", [])
        }

        os.makedirs("data/dashboard", exist_ok=True)
        with open("data/dashboard/latest_stats.json", "w") as f:
            json.dump(dashboard_state, f, indent=4)
        
        logging.info("✅ Final Phase Complete: Dashboard JSON updated.")
        
        duration = datetime.now() - start_time
        logging.info(f"⏱️ Pipeline Execution Time: {duration}")
        
        return dashboard_state # Automatically sent back to the Telegram Bot

    except Exception as e:
        logging.error(f"❌ PIPELINE FAILED: {str(e)}")
        return None

if __name__ == "__main__":
    # This allows you to run the whole thing from the terminal via `python pipeline.py`
    run_full_pipeline()