import asyncio
import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import html
from pipeline import run_full_pipeline

# Setup logging
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ALLOWED_USER_ID"))

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID:
        return
    await update.message.reply_text("🤖 *Quant Engine Online*\nSend /analyze to run the full pipeline.", parse_mode="Markdown")

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Triggers the full decoupled pipeline and returns the narrative."""
    if update.effective_user.id != ALLOWED_USER_ID:
        return

    status_msg = await update.message.reply_text("⏳ <b>Pipeline Initiated</b> (Est. 50s)...\nProcessing math & news.", parse_mode="HTML")

    try:
        # 1. TRIGGER THE FULL PIPELINE
        state = await asyncio.to_thread(run_full_pipeline)

        if state:
            # 2. EXTRACT DATA
            # We use .get() for safety and escape the narrative to prevent HTML breakage
            raw_narrative = state.get("narrative", "No analysis generated.")
            
            # Escape HTML characters (<, >, &) so the LLM doesn't accidentally trigger tags
            safe_narrative = html.escape(raw_narrative)
            
            regime = state["prediction"]["regime"]
            rsi = state["indicators"]["rsi"]
            fear = state["indicators"]["fear_greed"]
            
            regime_map = {0: "🔴 Bearish", 1: "🟡 Neutral", 2: "🟢 Bullish"}
            regime_icon = regime_map.get(regime, f"Unknown ({regime})")

            # 3. FORMAT OUTPUT (Using HTML tags instead of Markdown)
            response_text = (
                f"🏦 <b>INSTITUTIONAL BRIEFING</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Forecast:</b> {regime_icon}\n"
                f"📉 <b>RSI:</b> {rsi}  |  😨 <b>Fear/Greed:</b> {fear}\n\n"
                f"📝 <b>Analysis:</b>\n{safe_narrative}\n\n"
                f"✅ <b>System Update Complete</b>"
            )

            await status_msg.edit_text(response_text, parse_mode="HTML")
        else:
            await status_msg.edit_text("❌ <b>Pipeline Error:</b> The engine failed to produce a report.")

    except Exception as e:
        logging.error(f"Telegram Error: {e}")
        # If the HTML still fails, fallback to plain text so you at least see the message
        try:
            await status_msg.edit_text(f"⚠️ <b>Critical Bot Error:</b>\n{str(e)}", parse_mode="HTML")
        except:
            await status_msg.edit_text(f"⚠️ Critical Bot Error (Plain Text):\n{str(e)}")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze))
    
    print("🚀 Bot is listening for /analyze command...")
    application.run_polling()

if __name__ == "__main__":
    main()