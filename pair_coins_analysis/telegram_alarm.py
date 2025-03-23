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
    pair_1 = None,
    pair_2 = None,
    corrlation = None,
    cointegration = None,
):
    try:
        message_parts = []
        message_parts.append("=== Pair Strategy Execution Update ===\n")

        # Format positions
        if pair_1 and pair_2:
            message_parts.append( f"🔹 Pair 1:{pair_1}")
            message_parts.append( f"🔹 Pair 2:{pair_2}")
            message_parts.append(f"⭐ correlation: {corrlation}")
            message_parts.append(f"⭐ cointegration: {cointegration}")
        else:
            message_parts.append("No data to display")
            
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



