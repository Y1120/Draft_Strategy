from telegram import Bot
from telegram.error import TelegramError
import asyncio
from typing import Optional

class TelegramNotifier:
    def __init__(self, bot_token: str, default_chat_id: Optional[str] = None):
        self.bot = Bot(bot_token)
        self.default_chat_id = default_chat_id
        self._event_loop = None

    async def send_message(self, message: str, chat_id: Optional[str] = None) -> bool:
        """
        Send a message through Telegram
        Args:
            message: The message to send
            chat_id: Optional chat ID. If not provided, uses the default_chat_id
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            target_chat_id = self.default_chat_id
            if not target_chat_id:
                raise ValueError("No chat_id provided and no default_chat_id set")
            
            await self.bot.send_message(
                chat_id=target_chat_id,
                text=message,
            )
            return True
        except TelegramError as e:
            print(f"Error sending telegram message: {str(e)}")
            return False
        
    def send_message_sync(self, message: str, chat_id: Optional[str] = None) -> bool:
        """
        Synchronous version of send_message with proper event loop handling
        """
        try:
            # Create a new event loop for this thread if one doesn't exist
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function and close the loop
            result = loop.run_until_complete(self.send_message(message, chat_id))
            return result
        except Exception as e:
            print(f"Error in send_message_sync: {str(e)}")
            return False