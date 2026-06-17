"""
Telegram Notification Module
Sends alerts to Telegram groups/channels
"""
import os
import logging
import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.connection import create_connection as _orig_create_connection
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def _ipv4_create_connection(address, *args, **kwargs):
    """Force IPv4 DNS resolution to work around broken IPv6 routing to api.telegram.org"""
    host, port = address
    # Resolve hostname to IPv4 only
    infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    if not infos:
        raise OSError(f"getaddrinfo failed for host {host!r}")
    # Use the first IPv4 address
    family, socktype, proto, canonname, sockaddr = infos[0]
    return _orig_create_connection(sockaddr, *args, **kwargs)


class IPv4HTTPAdapter(HTTPAdapter):
    """HTTP adapter that forces IPv4 connections"""
    def send(self, *args, **kwargs):
        import urllib3.util.connection as urllib3_cn
        old_create = urllib3_cn.create_connection
        urllib3_cn.create_connection = _ipv4_create_connection
        try:
            return super().send(*args, **kwargs)
        finally:
            urllib3_cn.create_connection = old_create


def _get_telegram_session():
    """Create a requests.Session that forces IPv4 for Telegram API calls"""
    s = requests.Session()
    adapter = IPv4HTTPAdapter()
    s.mount("https://api.telegram.org", adapter)
    s.mount("http://api.telegram.org", adapter)
    return s

# Telegram configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("bot_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("chat_id")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Optional: Disable Telegram notifications via environment variable
TELEGRAM_DISABLED = os.getenv("TELEGRAM_DISABLED", "").lower() in ("true", "1", "yes")


class TelegramNotifier:
    """Centralized Telegram notification handler"""
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat/group ID (defaults to TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.enabled = bool(self.bot_token and self.chat_id) and not TELEGRAM_DISABLED
        self.session = _get_telegram_session()  # IPv4-only session for Telegram API
        
        if not self.enabled:
            if not self.bot_token:
                logger.warning("⚠️ Telegram notifications DISABLED: TELEGRAM_BOT_TOKEN not set")
            elif not self.chat_id:
                logger.warning("⚠️ Telegram notifications DISABLED: TELEGRAM_CHAT_ID not set")
            elif TELEGRAM_DISABLED:
                logger.info("ℹ️ Telegram notifications DISABLED: TELEGRAM_DISABLED=true")
            else:
                logger.warning("⚠️ Telegram notifications DISABLED: Unknown reason")
        else:
            logger.info(f"✅ Telegram notifier initialized successfully (chat_id: {self.chat_id[:10]}...)")
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message to Telegram
        
        Args:
            text: Message text (supports HTML formatting)
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            resp = self.session.post(url, data=data, timeout=10)
            
            if resp.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"❌ Telegram API error: {resp.status_code} - {resp.text}")
                # Log response details for debugging
                try:
                    error_data = resp.json()
                    logger.error(f"   Error details: {error_data}")
                except:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_photo(self, photo_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Send a photo to Telegram
        
        Args:
            photo_path: Path to image file
            caption: Optional caption text
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Check if file exists
            if not os.path.exists(photo_path):
                logger.warning(f"Photo file not found: {photo_path}")
                return False
            
            # Check file size (Telegram limit: 10MB for photos)
            file_size = os.path.getsize(photo_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"Photo file too large ({file_size / 1024 / 1024:.1f}MB), sending as document instead")
                return self.send_document(photo_path, caption=caption, parse_mode=parse_mode)
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption[:1024] if len(caption) > 1024 else caption,  # Telegram caption limit: 1024 chars
                    "parse_mode": parse_mode
                }
                resp = self.session.post(url, files=files, data=data, timeout=30)
            
            if resp.status_code == 200:
                logger.info(f"✅ Telegram photo sent successfully: {os.path.basename(photo_path)} ({file_size / 1024:.1f}KB)")
                return True
            else:
                logger.warning(f"Telegram API error: {resp.status_code} - {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {e}", exc_info=True)
            return False
    
    def send_document(self, document_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Send a document (GIF, video, etc.) to Telegram
        
        Args:
            document_path: Path to document file
            caption: Optional caption text
            parse_mode: 'HTML' or 'Markdown'
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Check if file exists
            if not os.path.exists(document_path):
                logger.warning(f"Document file not found: {document_path}")
                return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"
            
            with open(document_path, 'rb') as doc:
                files = {'document': doc}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": parse_mode
                }
                resp = self.session.post(url, files=files, data=data, timeout=30)
            
            if resp.status_code == 200:
                file_size = os.path.getsize(document_path)
                logger.info(f"✅ Telegram document sent successfully: {os.path.basename(document_path)} ({file_size / 1024:.1f}KB)")
                return True
            else:
                logger.warning(f"Telegram API error: {resp.status_code} - {resp.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram document: {e}", exc_info=True)
            return False
    
    def send_video(self, video_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Send a video file to Telegram using the sendVideo API.

        Args:
            video_path: Path to video file (MP4)
            caption: Optional caption text
            parse_mode: 'HTML' or 'Markdown'

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                return False

            file_size = os.path.getsize(video_path)
            # Telegram video limit: 50MB
            if file_size > 50 * 1024 * 1024:
                logger.warning(f"Video file too large ({file_size / 1024 / 1024:.1f}MB), sending as document instead")
                return self.send_document(video_path, caption=caption, parse_mode=parse_mode)

            url = f"https://api.telegram.org/bot{self.bot_token}/sendVideo"

            with open(video_path, 'rb') as video:
                files = {'video': video}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption[:1024] if len(caption) > 1024 else caption,
                    "parse_mode": parse_mode,
                    "supports_streaming": True
                }
                resp = self.session.post(url, files=files, data=data, timeout=120)

            if resp.status_code == 200:
                logger.info(f"✅ Telegram video sent successfully: {os.path.basename(video_path)} ({file_size / 1024:.1f}KB)")
                return True
            else:
                logger.warning(f"Telegram sendVideo API error: {resp.status_code} - {resp.text}")
                # Fallback: try sending as document
                logger.info("Retrying as document...")
                return self.send_document(video_path, caption=caption, parse_mode=parse_mode)

        except Exception as e:
            logger.error(f"Error sending Telegram video: {e}", exc_info=True)
            return False

    def send_alert(
        self,
        channel_id: str,
        alert_type: str,
        alert_message: str,
        image_path: Optional[str] = None,
        alert_data: Optional[Dict[str, Any]] = None,
        store_name: Optional[str] = None
    ) -> bool:
        """
        Send a formatted alert to Telegram
        
        Args:
            channel_id: Channel identifier
            alert_type: Type of alert (e.g., 'fall_alert', 'smoking_alert')
            alert_message: Alert message text
            image_path: Optional path to image/GIF file
            alert_data: Optional additional alert data
            store_name: Optional name of the store
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Format alert message
            alert_emoji = self._get_alert_emoji(alert_type)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build message
            message_parts = [
                f"{alert_emoji} <b>Alert: {self._format_alert_type(alert_type)}</b>"
            ]
            
            if store_name:
                message_parts.append(f"🏪 <b>Store:</b> {store_name}")
                
            message_parts.extend([
                f"📍 <b>Channel:</b> {channel_id}",
                f"⏰ <b>Time:</b> {timestamp}",
                f"",
                f"📝 <b>Message:</b> {alert_message}"
            ])
            
            # Add additional data if available
            if alert_data:
                if isinstance(alert_data, dict):
                    # Add relevant fields from alert_data
                    if 'detection_count' in alert_data:
                        message_parts.append(f"🔢 <b>Detections:</b> {alert_data['detection_count']}")
                    if 'crowd_count' in alert_data:
                        message_parts.append(f"👥 <b>Crowd Count:</b> {alert_data['crowd_count']}")
                        if 'threshold' in alert_data:
                            message_parts.append(f"📊 <b>Threshold:</b> {alert_data['threshold']}")
                    if 'raw_count' in alert_data:
                        message_parts.append(f"📈 <b>Raw Count:</b> {alert_data['raw_count']}")
                    if 'long_stay_count' in alert_data:
                        message_parts.append(f"⏱️ <b>Long Stay Count:</b> {alert_data['long_stay_count']}")
                    if 'duration' in alert_data:
                        duration = alert_data['duration']
                        if isinstance(duration, (int, float)):
                            message_parts.append(f"⏱️ <b>Duration:</b> {duration:.0f}s")
                    if 'violations' in alert_data:
                        violations = alert_data['violations']
                        if isinstance(violations, list):
                            violations = ", ".join(violations)
                        message_parts.append(f"⚠️ <b>Violations:</b> {violations}")
                    if 'waiting_time' in alert_data:
                        wait_min = alert_data['waiting_time'] / 60
                        message_parts.append(f"⏱️ <b>Wait Time:</b> {wait_min:.1f} min")
                    if 'order_wait_time' in alert_data and alert_data['order_wait_time'] is not None:
                        order_wait_min = alert_data['order_wait_time'] / 60
                        message_parts.append(f"📝 <b>Order Wait:</b> {order_wait_min:.1f} min")
                    if 'service_wait_time' in alert_data and alert_data['service_wait_time'] is not None:
                        service_wait_min = alert_data['service_wait_time'] / 60
                        message_parts.append(f"🍽️ <b>Service Wait:</b> {service_wait_min:.1f} min")
                    if 'table_id' in alert_data:
                        table_display = alert_data.get('table_name') or alert_data['table_id']
                        table_num = alert_data.get('table_number', '')
                        # Avoid a redundant "(#X)" when the number is identical to the display value
                        if table_num and str(table_num) != str(table_display):
                            message_parts.append(f"🪑 <b>Table:</b> {table_display} (#{table_num})")
                        else:
                            message_parts.append(f"🪑 <b>Table:</b> {table_display}")
                        # Include the table's ROI so reviewers can locate the table in the frame
                        roi_bbox = alert_data.get('roi_bbox')
                        if isinstance(roi_bbox, (list, tuple)) and len(roi_bbox) == 4:
                            x1, y1, x2, y2 = roi_bbox
                            message_parts.append(
                                f"📐 <b>Table ROI:</b> x: {float(x1):.2f}–{float(x2):.2f}, "
                                f"y: {float(y1):.2f}–{float(y2):.2f} (normalized)"
                            )
                    if 'violation_type' in alert_data:
                        violation_type = alert_data['violation_type']
                        if violation_type == 'order_wait':
                            message_parts.append(f"⚠️ <b>Type:</b> Order Wait Violation")
                        elif violation_type == 'service_wait':
                            message_parts.append(f"⚠️ <b>Type:</b> Service Wait Violation")
            
            message = "\n".join(message_parts)
            
            # Only send alert if an image/GIF/snapshot is available
            if image_path:
                # Try to resolve the actual file path
                resolved_path = self._resolve_image_path(image_path)
                
                if resolved_path and os.path.exists(resolved_path):
                    logger.info(f"Sending Telegram alert with image: {resolved_path}")
                    # Determine file type: GIF, video, or image
                    file_ext = Path(resolved_path).suffix.lower()
                    if file_ext == '.gif':
                        return self.send_document(resolved_path, caption=message)
                    elif file_ext in ('.mp4', '.avi', '.mkv', '.mov'):
                        return self.send_video(resolved_path, caption=message)
                    else:
                        # Send as photo for better display (jpg, png, etc.)
                        return self.send_photo(resolved_path, caption=message)
                else:
                    logger.warning(f"Skipping Telegram alert - image file not found: {image_path} (resolved: {resolved_path})")
                    return False
            else:
                # No image/GIF/snapshot provided - skip sending text-only alert
                logger.debug("Skipping Telegram alert - no image/GIF/snapshot provided")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
            return False
    
    def _get_alert_emoji(self, alert_type: str) -> str:
        """Get emoji for alert type"""
        emoji_map = {
            'fall_alert': '🚨',
            'smoking_alert': '🚬',
            'fire_smoke_alert': '🔥',
            'person_smoking_alert': '🚬',
            'unauthorized_entry_alert': '🚫',
            'queue_alert': '👥',
            'people_alert': '👥',
            'crowd_alert': '👥',
            'cash_alert': '💰',
            'dresscode_alert': '👔',
            'ppe_alert': '🦺',
            'grooming_alert': '💇',
            'queue_violation': '⚠️',
            'table_service_violation': '🍽️',
            'table_cleanliness_violation': '🧹',
            'material_theft_alert': '📦',
            'phone_alert': '📱',
            'mopping_alert': '🧹',
            'restricted_area_alert': '🚧'
        }
        return emoji_map.get(alert_type, '⚠️')
    
    def _format_alert_type(self, alert_type: str) -> str:
        """Format alert type for display"""
        type_map = {
            'fall_alert': 'Fall Detection',
            'smoking_alert': 'Smoking Detection',
            'fire_smoke_alert': 'Fire/Smoke Detection',
            'person_smoking_alert': 'Person Smoking',
            'unauthorized_entry_alert': 'Unauthorized Entry',
            'queue_alert': 'Queue Alert',
            'people_alert': 'People Alert',
            'crowd_alert': 'Crowd Detection',
            'cash_alert': 'Cash Detection',
            'dresscode_alert': 'Dress Code Violation',
            'ppe_alert': 'PPE Violation',
            'grooming_alert': 'Grooming Violation',
            'queue_violation': 'Queue Violation',
            'table_service_violation': 'Table Service Violation',
            'table_cleanliness_violation': 'Table Cleanliness Violation',
            'material_theft_alert': 'Material Theft',
            'phone_alert': 'Phone Usage',
            'mopping_alert': 'Mopping Detection',
            'restricted_area_alert': 'Restricted Area Violation'
        }
        return type_map.get(alert_type, alert_type.replace('_', ' ').title())
    
    def _resolve_image_path(self, image_path: str) -> Optional[str]:
        """
        Resolve image path by trying multiple possible locations
        
        Args:
            image_path: Original image path (can be relative or absolute)
            
        Returns:
            Resolved absolute path if file exists, None otherwise
        """
        if not image_path:
            return None
        
        # If already absolute and exists, return it
        if os.path.isabs(image_path) and os.path.exists(image_path):
            return image_path
        
        # Try different path variations
        # Only prepend "static/" if the path doesn't already start with it
        if image_path.startswith('static/') or image_path.startswith('static\\'):
            static_prefixed = image_path
        else:
            static_prefixed = os.path.join("static", image_path)
        possible_paths = [
            image_path,  # Original path
            static_prefixed,  # Relative to static (without double-prefix)
            os.path.abspath(image_path),  # Absolute from current dir
            os.path.abspath(static_prefixed),  # Absolute static path
        ]
        
        # Also try with just the filename in common directories
        filename = os.path.basename(image_path)
        static_dirs = [
            "static/alerts",
            "static/fall_snapshots",
            "static/cash_snapshots",
            "static/service_discipline",
            "static/smoking_snapshots",
            "static/phone_snapshots",
            "static/dresscode_snapshots",
            "static/ppe_snapshots",
            "static/grooming_snapshots",
            "static/table_service_violations",
            "static/table_cleanliness_violations",
            "static/mopping_snapshots",
            "static/restricted_area_snapshots",
        ]
        
        for static_dir in static_dirs:
            possible_paths.append(os.path.join(static_dir, filename))
            possible_paths.append(os.path.abspath(os.path.join(static_dir, filename)))
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                logger.debug(f"Resolved image path: {image_path} -> {path}")
                return path
        
        logger.warning(f"Could not resolve image path: {image_path}")
        return None


# Global instance
_telegram_notifier = None

def get_telegram_notifier() -> TelegramNotifier:
    """Get or create global Telegram notifier instance"""
    global _telegram_notifier
    if _telegram_notifier is None:
        _telegram_notifier = TelegramNotifier()
    return _telegram_notifier

