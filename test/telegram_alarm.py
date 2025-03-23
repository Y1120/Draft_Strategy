import time
import asyncio
import platform
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_init import TelegramNotifier 
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
LOG_FILE = os.path.join('logs', 'bot.log')
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 5  # Keep 5 backup files
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

async def send_telegram_messages(telegram: TelegramNotifier, messages: list[str]):
    """Send messages in chunks to avoid Telegram length limits"""
    for message in messages:
        try:
            await telegram.send_message(message)
            # Small delay between messages to avoid rate limiting
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error sending telegram message: {str(e)}")

def chunk_message(full_message: str, chunk_size: int = 4000) -> list[str]:
    """Split a long message into chunks that Telegram can handle"""
    messages = []
    lines = full_message.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in lines:
        if current_length + len(line) + 1 > chunk_size:  # +1 for newline
            messages.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_length = len(line)
        else:
            current_chunk.append(line)
            current_length += len(line) + 1
    
    if current_chunk:
        messages.append('\n'.join(current_chunk))
    
    return messages

async def fetch_data(
    telegram: TelegramNotifier = None,
    virtual_order = None,
    ai_order = None,
    reversed_order_vr = None,
    reversed_order_ai = None,
    position = None,
    gross_pnl = None,
    correlation = None,
    range = None,
    ratio = None
):

    try:
        message_parts = []
        message_parts.append("=== Pair Info Update ===\n")
        message_parts.append("Last One Month Correlation: " + str(correlation) )
        message_parts.append("Normal Range of Ratio(pct_change): "+ range)
        message_parts.append("Current Ratio: "+ str(ratio) + "\n")
                             
        message_parts.append("=== Pair Strategy Execution Update ===\n")
        # Format positions
        if position:
            message_parts.append("⭐ Current Positions:")
            
            # AI16Z Position
            if position.get("ai16z"):
                if position['ai16z'][2] != 0:
                    message_parts.append("🔹 AI16Z Position:")
                    message_parts.append(f"Symbol: {position['ai16z'][0]}")
                    message_parts.append(f"Side: {position['ai16z'][1]}")
                    message_parts.append(f"Amount: {position['ai16z'][2]}")
                    message_parts.append(f"Entry Price: {position['ai16z'][3]}")
                    message_parts.append(f"unrealized pnl: {position['ai16z'][4]}")
                else:
                    message_parts.append("🔹 AI16Z Position: No Position")

                # Virtual Position    
                if position['virtual'][2] != 0:
                    message_parts.append("🔹 Virtual Position:")
                    message_parts.append(f"Symbol: {position['virtual'][0]}")
                    message_parts.append(f"Side: {position['virtual'][1]}")
                    message_parts.append(f"Amount: {position['virtual'][2]}")
                    message_parts.append(f"Entry Price: {position['virtual'][3]}")
                    message_parts.append(f"unrealized pnl: {position['virtual'][4]}"+ "\n")
                else:
                    message_parts.append("🔹 Virtual Position: No Position"+ "\n")
            
        else:
            message_parts.append("⭐ No Positions"+ "\n")
        # Format Virtual Order
        if virtual_order:
            message_parts.append("⭐ Entering Positions:")
            message_parts.append("🔹 Virtual Order:")
            message_parts.append(f"Symbol: {virtual_order[0]}")
            message_parts.append(f"Side: {virtual_order[1]}")
            message_parts.append(f"Price: {virtual_order[2]}")
            message_parts.append(f"Amount: {virtual_order[3]}")
            message_parts.append(f"Status: {virtual_order[4]}")
            message_parts.append("🔹 AI16Z Order:")
            message_parts.append(f"Symbol: {ai_order[0]}")
            message_parts.append(f"Side: {ai_order[1]}")
            message_parts.append(f"Price: {ai_order[2]}")
            message_parts.append(f"Amount: {ai_order[3]}")
            message_parts.append(f"Status: {ai_order[4]}"+ "\n")


        if reversed_order_vr:
            message_parts.append("⭐ Reversed Positions:")
            message_parts.append("🔹 Virtual Reversed Order:")
            message_parts.append(f"Symbol: {reversed_order_vr[0]}")
            message_parts.append(f"Side: {reversed_order_vr[1]}")
            message_parts.append(f"Price: {reversed_order_vr[2]}")
            message_parts.append(f"Amount: {reversed_order_vr[3]}")
            message_parts.append(f"Status: {reversed_order_vr[4]}")
            
            message_parts.append("🔹 AI16Z Reversed Order:")
            message_parts.append(f"Symbol: {reversed_order_ai[0]}")
            message_parts.append(f"Side: {reversed_order_ai[1]}")
            message_parts.append(f"Price: {reversed_order_ai[2]}")
            message_parts.append(f"Amount: {reversed_order_ai[3]}")
            message_parts.append(f"Status: {reversed_order_ai[4]}" + "\n")

            
        # Format Gross PnL
        message_parts.append(f"❤ Gross PnL: {gross_pnl}")
            
        message = "\n".join(message_parts)
        if telegram:
            await telegram.send_message(message)
        return message
        
    except Exception as e:
        logger.error(f"Error in fetch_data: {e}")
        raise

async def cleanup():
    """Cleanup function to be called before shutdown"""
    for task in asyncio.all_tasks():
        task.cancel()
    await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)



