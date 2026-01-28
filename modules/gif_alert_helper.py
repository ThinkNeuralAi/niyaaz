"""
GIF Alert Helper Module
Provides easy-to-use functions for modules to save alerts as GIFs
"""

import logging
from datetime import datetime
from pathlib import Path
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class GifAlertHelper:
    """
    Helper class to easily add GIF alert support to any module
    
    Usage:
        helper = GifAlertHelper(channel_id, db_manager, app, socketio)
        helper.initialize_gif_recorder()
        
        # In process_frame:
        helper.add_frame_to_buffer(frame)
        
        # When alert triggers:
        helper.start_alert_recording(alert_type, alert_message, frame)
        
        # Continue in process_frame:
        if helper.is_recording():
            helper.add_alert_frame(frame)
            if helper.is_recording_complete():
                gif_info = helper.get_completed_gif()
                helper.save_alert_to_database(gif_info, alert_data)
    """
    
    def __init__(self, channel_id, db_manager=None, app=None, socketio=None):
        """
        Initialize GIF alert helper
        
        Args:
            channel_id: Channel identifier
            db_manager: Database manager instance
            app: Flask app instance
            socketio: SocketIO instance
        """
        self.channel_id = channel_id
        self.db_manager = db_manager
        self.app = app
        self.socketio = socketio
        self.gif_recorder = None
        self.last_gif_info = None
        
    def initialize_gif_recorder(self, buffer_size=90, gif_duration=3.0, fps=30):
        """
        Initialize the GIF recorder
        
        Args:
            buffer_size: Number of frames to keep in buffer (default: 90 = 3 seconds at 30fps)
            gif_duration: Duration of GIF in seconds (default: 3.0)
            fps: Target FPS for recording (default: 30)
        """
        self.gif_recorder = AlertGifRecorder(
            buffer_size=buffer_size,
            gif_duration=gif_duration,
            fps=fps
        )
        logger.info(f"[{self.channel_id}] GIF Alert Helper initialized")
    
    def add_frame_to_buffer(self, frame):
        """Add frame to GIF recorder buffer (call this in process_frame)"""
        if self.gif_recorder and frame is not None:
            self.gif_recorder.add_frame(frame)
    
    def start_alert_recording(self, alert_type, alert_message, frame, alert_data=None):
        """
        Start recording a GIF for an alert
        
        Args:
            alert_type: Type of alert (e.g., 'service_discipline_alert', 'table_cleanliness_alert')
            alert_message: Alert message text
            frame: Current frame to start recording with
            alert_data: Optional additional alert data
        """
        if not self.gif_recorder:
            logger.warning(f"[{self.channel_id}] GIF recorder not initialized")
            return False
        
        alert_info = {
            'message': alert_message,
            'channel_id': self.channel_id,
            'alert_type': alert_type,
            'alert_data': alert_data
        }
        
        self.gif_recorder.start_alert_recording(alert_info)
        if frame is not None:
            self.gif_recorder.add_alert_frame(frame)
        
        logger.info(f"[{self.channel_id}] Started GIF recording for {alert_type}")
        return True
    
    def add_alert_frame(self, frame):
        """Add frame during alert recording (call this in process_frame while recording)"""
        if self.gif_recorder and self.gif_recorder.is_recording_alert and frame is not None:
            self.gif_recorder.add_alert_frame(frame)
    
    def is_recording(self):
        """Check if currently recording an alert GIF"""
        return self.gif_recorder and self.gif_recorder.is_recording_alert
    
    def is_recording_complete(self):
        """Check if recording just completed"""
        if not self.gif_recorder:
            return False
        
        # Check if recording just finished
        if not self.gif_recorder.is_recording_alert and self.gif_recorder.alert_end_time:
            # Get the GIF info
            self.last_gif_info = self.gif_recorder.get_last_gif_info()
            if self.last_gif_info:
                return True
            # If get_last_gif_info returns None, try to get from stop_alert_recording
            # This handles the case where recording completed automatically
            return False
        
        return False
    
    def stop_and_get_gif(self):
        """Manually stop recording and get GIF info"""
        if not self.gif_recorder or not self.gif_recorder.is_recording_alert:
            return None
        
        gif_info = self.gif_recorder.stop_alert_recording()
        if gif_info:
            self.last_gif_info = gif_info
        return gif_info
    
    def get_completed_gif(self):
        """Get the completed GIF info (call after is_recording_complete returns True)"""
        return self.last_gif_info
    
    def save_alert_to_database(self, gif_info, alert_type, alert_message, alert_data=None):
        """
        Save alert GIF to database
        
        Args:
            gif_info: GIF info dictionary from get_completed_gif()
            alert_type: Type of alert
            alert_message: Alert message
            alert_data: Optional additional alert data
        """
        if not gif_info or not self.db_manager:
            logger.warning(f"[{self.channel_id}] Cannot save alert: gif_info or db_manager missing")
            return None
        
        try:
            if self.app:
                with self.app.app_context():
                    gif_id = self.db_manager.save_alert_gif(
                        channel_id=self.channel_id,
                        alert_type=alert_type,
                        gif_info=gif_info,
                        alert_message=alert_message,
                        alert_data=alert_data
                    )
            else:
                gif_id = self.db_manager.save_alert_gif(
                    channel_id=self.channel_id,
                    alert_type=alert_type,
                    gif_info=gif_info,
                    alert_message=alert_message,
                    alert_data=alert_data
                )
            
            logger.info(f"[{self.channel_id}] Alert GIF saved to database: ID {gif_id}")
            
            # Emit socket event if available
            if self.socketio:
                self.socketio.emit('alert_gif_created', {
                    'gif_id': gif_id,
                    'channel_id': self.channel_id,
                    'alert_type': alert_type,
                    'gif_filename': gif_info.get('gif_filename'),
                    'gif_url': f"/static/alerts/{gif_info.get('gif_filename')}",
                    'alert_message': alert_message,
                    'created_at': datetime.now().isoformat()
                })
            
            return gif_id
            
        except Exception as e:
            logger.error(f"[{self.channel_id}] Error saving alert GIF to database: {e}", exc_info=True)
            return None
    
    def get_gif_path_for_violation(self, gif_info):
        """
        Get GIF path to use in violation tables (snapshot_path field)
        This allows GIFs to be stored in the same field that snapshots used
        
        Args:
            gif_info: GIF info dictionary
            
        Returns:
            Path suitable for snapshot_path field (e.g., "static/alerts/alert_20260127_103000.gif")
            or relative path depending on what the violation table expects
        """
        if not gif_info or not gif_info.get('gif_path'):
            return None
        
        gif_path = gif_info['gif_path']
        
        # Return the path as-is - it should work in snapshot_path fields
        # Dashboard will display it correctly (browsers support GIFs)
        return gif_path
    
    def get_snapshot_path_for_violation(self, gif_info):
        """
        Alias for get_gif_path_for_violation - returns GIF path for snapshot_path field
        Use this when saving violations - the GIF path goes in snapshot_path field
        """
        return self.get_gif_path_for_violation(gif_info)
