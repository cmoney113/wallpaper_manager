#!/usr/bin/env python3
import os
import time
import random
import subprocess
import sys
import json
import shutil
import yaml
import uuid
import requests
import re
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, time as dt_time
from PIL import Image

# These imports are already included above
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QSpinBox, QSystemTrayIcon, QMenu, QStyle,
    QListWidget, QListWidgetItem, QMessageBox, QGroupBox, QTabWidget,
    QComboBox, QCheckBox, QTimeEdit, QSplitter,
    QProgressBar, QDialog, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QListView, QHeaderView,
    QAbstractItemView, QStackedWidget, QToolButton, QLineEdit,
    QStatusBar, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize,
    QDateTime, QSortFilterProxyModel, QTime, QEvent
)
from PyQt5.QtGui import (
    QIcon, QPalette, QColor, QPixmap
)

import platform
import asyncio
import aiohttp
import imagehash
from collections import defaultdict
import logging
import platform
import sys
import threading
import requests
import uuid
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Union, Tuple
import random
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

# Helper function to check if a PyQt object has been deleted
def sip_is_deleted(obj):
    try:
        # Try to access a property to see if object exists
        obj.objectName()
        return False  # Object exists
    except RuntimeError:
        return True  # Object has been deleted
    except Exception as e:
        logger.error(f"Error in sip_is_deleted: {e}")
        return True  # Assume deleted on error

# This function has been moved into the MainWindow class (see populate_views_with_data method)
# Keeping a small stub here for backward compatibility
def populate_views_with_data(self, data):
    if hasattr(self, 'populate_views_with_data') and callable(self.populate_views_with_data):
        self.populate_views_with_data(data)
    else:
        logger.error("Method populate_views_with_data not available in the current context")


# Function to handle exceptions and optionally show an error dialog
def handle_exception(context, exception, log_level=logging.ERROR, show_dialog=False, reraise=False):
    error_message = f"Error in {context}: {str(exception)}"
    
    # Log the error with appropriate level
    if log_level == logging.DEBUG:
        logger.debug(error_message)
    elif log_level == logging.INFO:
        logger.info(error_message)
    elif log_level == logging.WARNING:
        logger.warning(error_message)
    else:
        logger.error(error_message)
    
    # If requested, show an error dialog (must be called from the main thread)
    if show_dialog and 'MainWindow' in globals():
        # This will be executed in the UI context if available
        QApplication.instance().postEvent(
            MainWindow, 
            QShowErrorEvent(error_message)
        )
    
    # Optionally re-raise the exception for further handling
    if reraise:
        raise exception
    
    return False

# Initialize logging
logger = logging.getLogger("WallpaperManager")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# Add file handler if needed
try:
    log_file = os.path.join(os.path.dirname(__file__), 'wallpaper_manager.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # File gets all log levels
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.debug("Logging initialized successfully")
except Exception as e:
    print(f"Warning: Could not set up file logging: {e}")
    # We can't use handle_exception yet since it's not defined until after this
    logger.warning(f"Could not set up file logging: {e}")

# Define a custom event for showing errors in the UI thread
class QShowErrorEvent(QEvent):
    """Custom event type for showing error dialogs safely from any thread."""
    EVENT_TYPE = QEvent.Type(QEvent.registerEventType())
    
    def __init__(self, error_message):
        super().__init__(self.EVENT_TYPE)
        self.error_message = error_message

###############################################################################
#                              ImageManager Class                             #
###############################################################################

class ImageManager:
    """
    Manages image processing, organization, favorites, and tags for wallpapers.
    
    This class is responsible for handling all aspects of wallpaper image management,
    including organization, metadata tracking, favorites, tags, and image quality
    assessment. It provides methods for detecting duplicates, optimizing images,
    and cleaning up the wallpaper directory.
    
    Attributes:
        directory (str): Path to the wallpaper directory
        usage_stats (dict): Dictionary tracking image usage statistics
        tags_data (dict): Dictionary storing tags for each image
        metadata (dict): Dictionary storing metadata for each image
        hash_cache (dict): Cache for image hashes to avoid recalculation
    """
    """Handles image processing, organization, favorites, and tags."""

    def __init__(self, directory):
        self.directory = directory
        
        # Load usage stats
        self.usage_stats = self.load_usage_stats()
        
        # Load tags data
        self.tags_data = self.load_tags_data()
        
        # Load metadata
        self.metadata = self.load_metadata()
        
        # Initialize hash cache for duplicate detection
        self.hash_cache = {}
    
    def load_usage_stats(self):
        """
        Load image usage statistics from the storage file.
        
        Returns:
            dict: A dictionary of usage statistics or empty defaultdict if not found
        """
        stats_file = Path(self.directory) / '.usage_stats.json'
        logger.debug(f"Loading usage stats from {stats_file}")
        try:
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                logger.debug(f"Successfully loaded usage stats for {len(stats)} images")
                return stats
            else:
                logger.debug("Usage stats file not found, returning empty stats")
                return defaultdict(int)
        except Exception as e:
            logger.error(f"Error loading usage stats: {e}")
            return defaultdict(int)
            
    def save_usage_stats(self):
        """
        Save image usage statistics to the storage file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        stats_file = Path(self.directory) / '.usage_stats.json'
        logger.debug(f"Saving usage stats to {stats_file}")
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.usage_stats, f)
            logger.debug(f"Successfully saved usage stats for {len(self.usage_stats)} images")
            return True
        except Exception as e:
            return handle_exception("save_usage_stats", e)
    def load_tags_data(self):
        """
        Load image tags data from the storage file.
        
        Returns:
            dict: A dictionary of image tags or empty dict if not found
        """
        tags_file = Path(self.directory) / '.tags.json'
        logger.debug(f"Loading tags data from {tags_file}")
        if tags_file.exists():
            try:
                with open(tags_file, 'r') as f:
                    tags = json.load(f)
                    logger.debug(f"Successfully loaded tags for {len(tags)} images")
                    return tags
            except Exception as e:
                return handle_exception("load_tags_data", e)
        logger.debug("No tags file found, creating new tags database")
        return {}

    def save_tags_data(self):
        tags_file = Path(self.directory) / '.tags.json'
        try:
            with open(tags_file, 'w') as f:
                json.dump(self.tags_data, f)
        except Exception as e:
            logger.error(f"Error saving tags data: {e}")

    def load_metadata(self):
        """Load image metadata from file."""
        metadata_file = Path(self.directory) / '.metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return {}

    def save_metadata(self):
        """Save image metadata to file."""
        metadata_file = Path(self.directory) / '.metadata.json'
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def add_image_metadata(self, file_path, new_metadata):
        """Add metadata for a newly downloaded image."""
        # First make sure the file exists
        if not os.path.exists(file_path):
            logger.error(f"Cannot add metadata - file does not exist: {file_path}")
            return False

        # Get the current metadata, if any
        current_data = self.metadata.get(file_path, {})

        # Update with new metadata
        current_data.update(new_metadata)

        # Save the metadata
        self.metadata[file_path] = current_data
        self.save_metadata()

        return True

    def get_image_hash(self, image_path):
        """Generate or retrieve image hash for duplicate detection."""
        if image_path in self.hash_cache:
            return self.hash_cache[image_path]
        try:
            with Image.open(image_path) as img:
                img_hash = str(imagehash.average_hash(img))
                self.hash_cache[image_path] = img_hash
                return img_hash
        except Exception as e:
            logger.error(f"Error generating hash for {image_path}: {e}")
            return None

    def find_duplicates(self):
        """Find duplicate images in the directory."""
        hash_dict = defaultdict(list)
        duplicates = []
        for file in Path(self.directory).glob('*'):
            if file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
                img_hash = self.get_image_hash(str(file))
                if img_hash:
                    hash_dict[img_hash].append(file)
        for hash_val, files in hash_dict.items():
            if len(files) > 1:
                duplicates.extend(files[1:])
        return duplicates

    def check_image_quality(self, image_path):
        """Check if image meets basic quality standards."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width < 1920 or height < 1080:
                    return False, "Resolution too low"
                if os.path.getsize(image_path) < 100 * 1024:
                    return False, "File size too small"
                return True, "OK"
        except Exception as e:
            return False, str(e)

    def optimize_image(self, image_path):
        """Optimize image for wallpaper use."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                max_size = (3840, 2160)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.LANCZOS)
                output_path = str(image_path).replace('.', '_optimized.')
                img.save(output_path, 'JPEG', quality=85, optimize=True)
                return output_path
        except Exception as e:
            logger.error(f"Error optimizing {image_path}: {e}")
            return None

    def cleanup_directory(self, max_files=None):
        """Clean up the wallpaper directory (duplicates, low quality, etc.)."""
        try:
            wallpapers = []
            for file in Path(self.directory).glob('*'):
                if file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
                    wallpapers.append((file, file.stat().st_mtime))

            # Sort by modification time, newest first
            wallpapers.sort(key=lambda x: x[1], reverse=True)

            # Remove duplicates
            duplicates = self.find_duplicates()
            for dup in duplicates:
                dup.unlink()

            # Remove low quality images
            for file, _ in wallpapers:
                quality_ok, _ = self.check_image_quality(str(file))
                if not quality_ok:
                    file.unlink()

            # Remove excess if max_files is specified
            if max_files and len(wallpapers) > max_files:
                for file, _ in wallpapers[max_files:]:
                    if file.exists():
                        file.unlink()

            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

    def record_usage(self, image_path):
        """Record when an image is used."""
        entry = self.usage_stats.get(str(image_path), {})
        entry['last_used'] = datetime.now().isoformat()
        entry['use_count'] = entry.get('use_count', 0) + 1
        self.usage_stats[str(image_path)] = entry
        self.save_usage_stats()

    # Favorites functionality
    def add_favorite(self, image_path):
        """Mark an image as favorite."""
        entry = self.usage_stats.get(str(image_path), {})
        entry['favorite'] = True
        self.usage_stats[str(image_path)] = entry
        self.save_usage_stats()

    def remove_favorite(self, image_path):
        """Remove favorite status from an image."""
        entry = self.usage_stats.get(str(image_path), {})
        entry['favorite'] = False
        self.usage_stats[str(image_path)] = entry
        self.save_usage_stats()

    def is_favorite(self, image_path):
        """Check if an image is marked as favorite."""
        entry = self.usage_stats.get(str(image_path), {})
        return entry.get('favorite', False)

    def get_favorites(self):
        """Return a list of (path, last_used) for all favorites."""
        favorites = []
        for path_str, data in self.usage_stats.items():
            if data.get('favorite'):
                last_used = data.get('last_used', 'Never')
                favorites.append((path_str, last_used))
        return favorites

    # Tags functionality
    def add_tags(self, image_path, tags):
        """Add tags to an image (comma-separated or list)."""
        if not isinstance(tags, list):
            # if user typed comma-separated
            tags = [t.strip() for t in tags.split(',')]

        path_str = str(image_path)
        current_tags = self.tags_data.get(path_str, [])
        current_tags.extend(tags)
        self.tags_data[path_str] = list(set(current_tags))  # remove duplicates
        self.save_tags_data()

    def remove_tag(self, image_path, tag):
        """Remove a tag from an image."""
        path_str = str(image_path)
        if path_str in self.tags_data:
            self.tags_data[path_str] = [t for t in self.tags_data[path_str] if t != tag]
            self.save_tags_data()

    def get_tags(self, image_path):
        """Get tags for an image."""
        return self.tags_data.get(str(image_path), [])

    def get_all_tags(self):
        """Get a list of all unique tags used."""
        all_tags = set()
        for tags in self.tags_data.values():
            all_tags.update(tags)
        return sorted(list(all_tags))

###############################################################################
#                          WallpaperWorker Class                              #
###############################################################################
from PyQt5.QtCore import QThread, pyqtSignal
class WallpaperWorker(QThread):
    """
    Worker thread for automatic wallpaper changing.
    
    This class runs in a separate thread and handles the automatic changing of 
    wallpapers based on configured intervals and active hours. It supports multiple
    platforms (Windows, macOS, Linux) and handles various desktop environments.
    
    Signals:
        wallpaperChanged(str): Emitted when wallpaper is changed, with the path
        statusUpdated(str): Emitted to provide status updates
        error(str): Emitted when an error occurs
    
    Attributes:
        config (dict): Configuration dictionary for wallpaper changing
        image_manager (ImageManager): Reference to the image manager
        is_running (bool): Flag to control thread execution
        monitors (list): List of detected monitors
        used_images (set): Set of recently used wallpaper paths
    """
    """Worker thread for wallpaper changing."""

    wallpaperChanged = pyqtSignal(str)
    statusUpdated = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config, image_manager):
        super().__init__()
        self.config = config
        self.image_manager = image_manager
        self.is_running = True
        from screeninfo import get_monitors
        self.monitors = get_monitors()
        self.used_images = set()

    def is_active_time(self):
        current_time = datetime.now().time()
        start_time = dt_time(self.config['active_hours_start'])
        end_time = dt_time(self.config['active_hours_end'])

        # Handle time spans that cross midnight properly
        if start_time <= end_time:
            # Normal time span within the same day
            return start_time <= current_time <= end_time
        else:
            # Time span crosses midnight
            return current_time >= start_time or current_time <= end_time

    def run(self):
        while self.is_running:
            try:
                if self.is_active_time():
                    self.change_wallpaper()
                for _ in range(self.config['interval']):
                    if not self.is_running:
                        return
                    time.sleep(1)
            except Exception as e:
                self.error.emit(str(e))
                return

    def change_wallpaper(self):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        directory = self.config['wallpaper_directory']
        wallpapers = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(valid_exts)
        ]
        if not wallpapers:
            self.error.emit("No valid wallpapers found in the directory.")
            return

        if self.config.get('display_all_before_repeat', False):
            if len(self.used_images) == len(wallpapers):
                self.statusUpdated.emit("All wallpapers displayed. Resetting cycle...")
                self.used_images.clear()

        remaining = list(set(wallpapers) - self.used_images)
        if not remaining:
            remaining = wallpapers
            self.used_images.clear()

        chosen = random.choice(remaining)
        try:
            if platform.system() == 'Darwin':
                for mon in self.monitors:
                    script = f'''
                    tell application "System Events"
                        tell every desktop
                            set picture to "{chosen}"
                        end tell
                    end tell
                    '''
                    subprocess.run(["osascript", "-e", script], check=True)
            elif platform.system() == 'Linux':
                # Ubuntu Budgie uses gsettings for changing wallpaper
                try:
                    # Check for Ubuntu Budgie
                    desktop_session = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
                    if 'budgie' in desktop_session:
                        # For Budgie desktop
                        subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri', f"file://{chosen}"], check=True)
                        # Also set the dark mode variant just in case
                        subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri-dark', f"file://{chosen}"], check=True)
                    else:
                        # Generic method for other Linux environments
                        desktop_env = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
                        if 'gnome' in desktop_env:
                            subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri', f"file://{chosen}"], check=True)
                        elif 'kde' in desktop_env:
                            script = f"""
                            var allDesktops = desktops();
                            for (var i=0; i<allDesktops.length; i++) {{
                                d = allDesktops[i];
                                d.wallpaperPlugin = "org.kde.image";
                                d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");
                                d.writeConfig("Image", "{chosen}");
                            }}
                            """
                            subprocess.run(['qdbus', 'org.kde.plasmashell', '/PlasmaShell', 'evaluateScript', script], check=True)
                        elif 'xfce' in desktop_env:
                            subprocess.run(['xfconf-query', '-c', 'xfce4-desktop', '-p', '/backdrop/screen0/monitor0/workspace0/last-image', '-s', chosen], check=True)
                        elif 'mate' in desktop_env:
                            subprocess.run(['gsettings', 'set', 'org.mate.background', 'picture-filename', chosen], check=True)
                        else:
                            # Try a more generic method for other desktops
                            subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri', f"file://{chosen}"], check=True)
                except Exception as e:
                    self.error.emit(f"Error setting wallpaper on Linux: {e}")
            elif platform.system() == 'Windows':
                import ctypes
                SPI_SETDESKWALLPAPER = 20
                ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, chosen, 3)
            self.used_images.add(chosen)
            self.image_manager.record_usage(chosen)
            self.wallpaperChanged.emit(chosen)
            self.statusUpdated.emit(f"Changed wallpaper to: {os.path.basename(chosen)}")
        except subprocess.CalledProcessError as e:
            self.error.emit(f"Error setting wallpaper: {e}")

    def stop(self):
        self.is_running = False


###############################################################################
#                             ScanWorker Class                                #
###############################################################################
class ScanWorker(QThread):
    """Worker thread for scanning wallpaper directory."""
    data_ready = pyqtSignal(list)

    def __init__(self, directory, valid_extensions, image_manager):
        super().__init__()
        self.directory = directory
        self.valid_extensions = valid_extensions
        self.image_manager = image_manager

    def run(self):
        data = []
        for fname in os.listdir(self.directory):
            if fname.lower().endswith(self.valid_extensions):
                path = os.path.join(self.directory, fname)
                try:
                    stat = os.stat(path)
                    size_kb = stat.st_size // 1024
                    date_modified = datetime.fromtimestamp(stat.st_mtime)
                    ext = Path(path).suffix.lower().lstrip('.')
                    data.append({
                        "name": fname,
                        "path": path,
                        "size_kb": size_kb,
                        "modified": date_modified,
                        "kind": ext,
                        "favorite": self.image_manager.is_favorite(path),
                        "tags": self.image_manager.get_tags(path)
                    })
                except Exception as e:
                    logger.error(f"Error reading file stats: {e}")

        # Sort data by name before emitting
        data.sort(key=lambda d: d["name"].lower())
        self.data_ready.emit(data)

###############################################################################
#                        MultiSourceScraperWorker Class                       #
###############################################################################
from PyQt5.QtCore import QMutex

class MultiSourceScraperWorker(QThread):
    """
    Worker thread for scraping wallpapers from multiple sources.
    
    This class handles the fetching and downloading of wallpapers from 
    various sources like Wallhaven and DuckDuckGo. It supports both preview 
    fetching and actual downloading of images.
    
    The class provides thread-safe operations using a mutex lock, proper resource
    cleanup, and cross-platform file and URL handling to ensure compatibility
    across different operating systems.
    """

    finished = pyqtSignal()
    statusUpdated = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    previewReady = pyqtSignal(list)  # Signal to emit when preview data is ready

    def __init__(self, config: Dict[str, Any], image_manager: 'ImageManager', preview_only: bool = False):
        """Initialize the scraper worker with configuration and image manager.
        
        Args:
            config: Dictionary containing scraper configuration
            image_manager: ImageManager instance for managing downloaded images
            preview_only: If True, only fetch image metadata without downloading
        """
        super().__init__()
        self.config = config
        self.image_manager = image_manager
        self.is_running = True
        self.preview_only = preview_only
        self.preview_data = []
        self.mutex = QMutex()
        
    def _ensure_directory(self, directory: Union[str, Path]) -> Path:
        """Ensure directory exists and return Path object."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
        
    @contextmanager
    def _thread_safe(self):
        """Context manager for thread-safe operations."""
        try:
            self.mutex.lock()
            yield
        finally:
            self.mutex.unlock()

    def run(self) -> None:
        """
        Main thread execution method that handles image fetching and downloading.
        
        This method manages all operations, using the thread-safe context manager
        to ensure proper mutex locking when accessing shared resources.
        """
        with self._thread_safe():
            try:
                # Create download directory if it doesn't exist
                download_dir = self._ensure_directory(self.config['wallpaper_directory'])

                # Get the enabled sources from config
                enabled_sources = self.config.get('enabled_sources', ['wallhaven', 'unsplash'])
                # Get the keywords from config, defaulting to some popular categories
                # Check for custom keyword first, if not available use the selected category
                custom_keyword = self.config.get('custom_keyword', '')
                if custom_keyword:
                    keyword = custom_keyword
                else:
                    keyword = self.config.get('scraper_category', 'landscapes')
                # Number of images to download
                max_images = self.config.get('scraper_limit', 10)
                total_downloaded = 0

                if self.preview_only:
                    self.statusUpdated.emit(f"Fetching image previews for: {keyword}")
                else:
                    self.statusUpdated.emit(f"Starting download with keyword: {keyword}")

                # Download from enabled sources
                for source in enabled_sources:
                    if not self.is_running:
                        break
                    if source == 'wallhaven':
                        if self.preview_only:
                            self._fetch_wallhaven_previews(keyword, max_images // len(enabled_sources))
                        else:
                            images_downloaded = self._download_from_wallhaven(download_dir, keyword, max_images // len(enabled_sources))
                            total_downloaded += images_downloaded
                    elif source == 'duckduckgo':
                        if self.preview_only:
                            self._fetch_duckduckgo_previews(keyword, max_images // len(enabled_sources))
                        else:
                            images_downloaded = self._download_from_duckduckgo(download_dir, keyword, max_images // len(enabled_sources))
                            total_downloaded += images_downloaded

                    # Update progress (assuming equal distribution across sources)
                    progress_value = min(100, int(100 * enabled_sources.index(source) / len(enabled_sources)))
                    self.progress.emit(progress_value)

                if self.preview_only:
                    self.statusUpdated.emit(f"Preview fetch complete. Found {len(self.preview_data)} images.")
                    self.previewReady.emit(self.preview_data)
                else:
                    self.statusUpdated.emit(f"Download complete. Downloaded {total_downloaded} wallpapers.")
                self.progress.emit(100)
            except Exception as e:
                self.error.emit(f"Error during download: {str(e)}")
            finally:
                self.finished.emit()
            
    def _fetch_wallhaven_previews(self, keyword: str, max_count: int) -> None:
        """
        Fetch image metadata from Wallhaven API without downloading.
        
        Args:
            keyword: Search keyword
            max_count: Maximum number of images to fetch
        """
        logger.info(f"Starting Wallhaven preview fetch for keyword: {keyword}")
        self.statusUpdated.emit(f"Fetching previews from Wallhaven with keyword: {keyword}")
        
        # Get resolution preference
        resolution = self.config.get('scraper_resolution', 'any')
        resolution_param = ""
        
        # Map resolution setting to Wallhaven API parameters
        if resolution != "Any":
            if resolution == "1920x1080":
                resolution_param = "&resolutions=1920x1080"
            elif resolution == "2560x1440":
                resolution_param = "&resolutions=2560x1440"
            elif resolution == "3840x2160":
                resolution_param = "&resolutions=3840x2160"
        
        # Apply purity filter
        purity_filter = self.config.get('purity_filter', 110)
        purity_param = ""
        if purity_filter == 100:  # SFW Only
            purity_param = "&purity=100"
        elif purity_filter == 110:  # SFW + Sketchy
            purity_param = "&purity=110"
        else:  # All (including NSFW)
            purity_param = "&purity=111"
        
        # Construct API URL
        search_query = keyword.replace(' ', '+')
        api_url = f"https://wallhaven.cc/api/v1/search?q={search_query}{resolution_param}{purity_param}&sorting=random"
        
        logger.debug(f"Wallhaven API URL: {api_url}")
        
        try:
            # Set request headers
            headers = {
                'User-Agent': 'Enhanced Wallpaper Manager/1.0',
                'Accept': 'application/json'
            }
            
            # Make API request
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            wallpapers = data.get('data', [])
            
            count = 0
            for wallpaper in wallpapers:
                if not self.is_running or count >= max_count:
                    break
                    
                try:
                    # Extract metadata
                    image_url = wallpaper.get('path')
                    thumbnail_url = wallpaper.get('thumbs', {}).get('small')
                    resolution = wallpaper.get('resolution')
                    file_size = wallpaper.get('file_size', 0)
                    tags = [tag.get('name') for tag in wallpaper.get('tags', [])]
                    
                    if not image_url or not thumbnail_url:
                        continue
                        
                    # Create a filename
                    image_id = wallpaper.get('id', str(uuid.uuid4()))
                    file_ext = os.path.splitext(image_url)[1]
                    if not file_ext:
                        file_ext = '.jpg'
                    
                    filename = f"wallhaven_{image_id}{file_ext}"
                    
                    # Download thumbnail for preview
                    thumbnail_data = None
                    try:
                        thumbnail_response = requests.get(thumbnail_url, headers=headers, timeout=5)
                        thumbnail_response.raise_for_status()
                        thumbnail_data = thumbnail_response.content
                    except Exception as e:
                        logger.error(f"Error downloading thumbnail: {e}")
                        continue
                    
                    # Add to preview data
                    with self._thread_safe():
                        self.preview_data.append({
                            'source': 'wallhaven',
                            'filename': filename,
                            'url': image_url,
                            'thumbnail_data': thumbnail_data,
                            'resolution': resolution,
                            'size_kb': file_size // 1024 if file_size else 0,
                            'tags': tags,
                        })
                        count += 1
                        
                except Exception as e:
                    self.error.emit(f"Error fetching preview from Wallhaven: {str(e)}")
                    
            self.statusUpdated.emit(f"Found {count} wallpapers from Wallhaven")
        except Exception as e:
            self.error.emit(f"Error fetching Wallhaven previews: {e}")
            return


    def _download_from_duckduckgo(self, download_dir: str, keyword: str, max_count: int) -> int:
        """
        Download images from DuckDuckGo image search.
        
        Args:
            download_dir: Directory to save downloaded images
            keyword: Search keyword
            max_count: Maximum number of images to download
            
        Returns:
            Number of successfully downloaded images
        """
        self.statusUpdated.emit(f"Downloading from DuckDuckGo with keyword: {keyword}")
        
        downloaded_count = 0
        
        # Construct search query (add "wallpaper" to keyword for better results)
        search_query = f"{keyword} wallpaper"
        # Apply purity filter
        purity_filter = self.config.get('purity_filter', 110)
        safe_search_param = ""
        if purity_filter == 100:  # SFW Only
            safe_search_param = "&kp=1"
        elif purity_filter == 110:  # SFW + Sketchy
            safe_search_param = "&kp=moderate"
        else:  # All (111)
            safe_search_param = "&kp=-1"
            
        # Create search URL using helper method for cross-platform URL handling
        search_url = self._create_search_url(search_query, safe_search_param)
        
        try:
            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://duckduckgo.com/',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # First, get the search results page
            response = requests.get(search_url, headers=headers)
            if response.status_code != 200:
                self.error.emit(f"DuckDuckGo search error: {response.status_code}")
                return 0
                
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # DuckDuckGo loads images via AJAX, so we need to find the vqd parameter
            vqd_match = re.search(r'vqd="([^"]+)"', response.text)
            if not vqd_match:
                vqd_match = re.search(r'vqd=([^&]+)&', response.text)
            
            if not vqd_match:
                self.error.emit("Couldn't extract vqd parameter from DuckDuckGo")
                return 0
                
            vqd = vqd_match.group(1)
            
            # Apply same purity filter to AJAX request - using proper URL encoding
            # Create AJAX URL with proper encoding
            ajax_url = self._create_search_url(
                f"{search_query}&o=json&p=1&s=100&u=bing&f=,,,&l=us-en&vqd={vqd}",
                safe_search_param
            )
            
            # Make the AJAX request to get the image data
            ajax_response = requests.get(ajax_url, headers=headers)
            
            if ajax_response.status_code != 200:
                self.error.emit(f"DuckDuckGo AJAX error: {ajax_response.status_code}")
                return 0
                
            # Parse the JSON response
            try:
                image_data = ajax_response.json()
                results = image_data.get('results', [])
                
                for i, result in enumerate(results):
                    if not self.is_running or downloaded_count >= max_count:
                        break
                        
                    try:
                        # Get image URL - prefer the higher quality 'image' URL over thumbnail
                        image_url = result.get('image')
                        if not image_url:
                            continue
                            
                        # Create a filename using a unique ID
                        image_id = result.get('id', str(uuid.uuid4()))
                        # Get file extension from URL or default to jpg
                        file_ext = os.path.splitext(image_url)[1]
                        if not file_ext:
                            file_ext = '.jpg'
                            
                        filename = f"duckduckgo_{image_id}{file_ext}"
                        # Use Path for cross-platform file handling
                        file_path = Path(download_dir) / filename
                        
                        # Skip if file already exists
                        if file_path.exists():
                            continue
                            
                        # Download the image
                        image_response = requests.get(image_url, stream=True, headers=headers)
                        if image_response.status_code == 200:
                            with self._thread_safe():
                                if not file_path.exists():
                                    with open(str(file_path), 'wb') as f:
                                        for chunk in image_response.iter_content(1024):
                                            f.write(chunk)
                            # Extract tags from the title and alt text
                            title = result.get('title', '')
                            tags = [tag.strip() for tag in title.split() if len(tag.strip()) > 3]
                            
                            # Register the image with image manager
                            # file_path is already a Path object, no need to convert again
                            self.image_manager.add_image_metadata(str(file_path), {
                                'source': 'duckduckgo',  # Add source field
                                'tags': tags,
                            })
                            
                            downloaded_count += 1
                            self.statusUpdated.emit(f"Downloaded {downloaded_count} images from DuckDuckGo")
                            
                    except Exception as e:
                        self.error.emit(f"Error downloading image from DuckDuckGo: {str(e)}")
                        
            except ValueError as e:
                self.error.emit(f"Error parsing JSON from DuckDuckGo: {str(e)}")
                return downloaded_count
            
        except Exception as e:
            self.error.emit(f"Error with DuckDuckGo search: {str(e)}")
            return downloaded_count
    def _fetch_duckduckgo_previews(self, keyword: str, max_count: int) -> None:
        """
        Fetch image metadata from DuckDuckGo image search without downloading.
        
        Args:
            keyword: Search keyword
            max_count: Maximum number of images to fetch
        """
        logger.info(f"Starting DuckDuckGo preview fetch for keyword: {keyword}")
        self.statusUpdated.emit(f"Fetching previews from DuckDuckGo with keyword: {keyword}")
        
        # Construct search query (add "wallpaper" to keyword for better results)
        search_query = f"{keyword} wallpaper"
        
        # Apply purity filter
        purity_filter = self.config.get('purity_filter', 110)
        safe_search_param = ""
        if purity_filter == 100:  # SFW Only
            safe_search_param = "&kp=1"
        elif purity_filter == 110:  # SFW + Sketchy
            safe_search_param = "&kp=moderate"
        else:  # All (111)
            safe_search_param = "&kp=-1"
            
        # Create search URL using helper method for cross-platform URL handling
        search_url = self._create_search_url(search_query, safe_search_param)
        logger.debug(f"DuckDuckGo search URL: {search_url}")
        
        try:
            # Set headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://duckduckgo.com/',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # First, get the search results page
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            logger.debug(f"DuckDuckGo response status code: {response.status_code}")
            
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # DuckDuckGo loads images via AJAX, so we need to find the vqd parameter
            vqd_match = re.search(r'vqd="([^"]+)"', response.text)
            if not vqd_match:
                vqd_match = re.search(r'vqd=([^&]+)&', response.text)
            
            if not vqd_match:
                self.error.emit("Couldn't extract vqd parameter from DuckDuckGo")
                return
                
            vqd = vqd_match.group(1)
            
            # Now make the AJAX request to get the image data
            ajax_url = self._create_search_url(
                f"{search_query}&o=json&p=1&s=100&u=bing&f=,,,&l=us-en&vqd={vqd}",
                safe_search_param
            )
            
            try:
                # Apply thread safety when making the request
                ajax_response = requests.get(ajax_url, headers=headers, timeout=10)
                ajax_response.raise_for_status()
                logger.debug(f"DuckDuckGo AJAX response status code: {ajax_response.status_code}")
                
                # Parse the JSON response
                try:
                    image_data = ajax_response.json()
                    results = image_data.get('results', [])
                    
                    count = 0
                    for result in results:
                        if not self.is_running or count >= max_count:
                            break
                            
                        try:
                            # Get image URL - prefer the higher quality 'image' URL over thumbnail
                            image_url = result.get('image')
                            thumbnail_url = result.get('thumbnail') or result.get('thumbnail_src') or result.get('image')
                            
                            if not image_url or not thumbnail_url:
                                continue
                                
                            # Create a filename using a unique ID
                            image_id = result.get('id', str(uuid.uuid4()))
                            # Get file extension from URL or default to jpg
                            file_ext = os.path.splitext(image_url)[1]
                            if not file_ext:
                                file_ext = '.jpg'
                                
                            filename = f"duckduckgo_{image_id}{file_ext}"
                            
                            # Download thumbnail for preview
                            thumbnail_data = None
                            try:
                                logger.debug(f"Thumbnail URL: {thumbnail_url}")
                                thumbnail_response = requests.get(thumbnail_url, headers=headers, timeout=5)
                                thumbnail_response.raise_for_status()
                                logger.debug(f"Thumbnail status code: {thumbnail_response.status_code}")
                                thumbnail_data = thumbnail_response.content
                                
                                # Extract metadata
                                width = result.get('width', 0)
                                height = result.get('height', 0)
                                title = result.get('title', '')
                                
                                # Extract tags from the title
                                tags = [tag.strip() for tag in title.split() if len(tag.strip()) > 3]
                                
                                # Add to preview data
                                with self._thread_safe():
                                    self.preview_data.append({
                                        'source': 'duckduckgo',
                                        'filename': filename,
                                        'url': image_url,
                                        'thumbnail_data': thumbnail_data,
                                        'resolution': f"{width}x{height}" if width and height else "Unknown",
                                        'size_kb': 0,  # Size unknown until download
                                        'tags': tags,
                                    })
                                    count += 1
                                    
                            except Exception as e:
                                logger.error(f"Error downloading thumbnail: {e}")
                                continue
                                
                        except Exception as e:
                            self.error.emit(f"Error fetching preview from DuckDuckGo: {str(e)}")
                            
                    self.statusUpdated.emit(f"Found {count} wallpapers from DuckDuckGo")
                    
                except ValueError as e:
                    logger.error(f"Error parsing JSON from DuckDuckGo: {str(e)}")
                    return
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"DuckDuckGo AJAX error: {e}")
                return
                
        except Exception as e:
            self.error.emit(f"Error fetching DuckDuckGo previews: {e}")
            return

    def stop(self) -> None:
        """Stop the scraper worker thread."""
        self.is_running = False
    def __del__(self) -> None:
        """Ensure proper cleanup of thread and resources."""
        try:
            self.stop()
            if hasattr(self, 'mutex'):
                self.mutex.unlock()  # Ensure mutex is unlocked
            if not self.wait(5000):  # 5 second timeout
                logger.warning("Thread cleanup timed out")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
    def _create_search_url(self, query: str, safe_search_param: str) -> str:
        """
        Create a properly encoded search URL for DuckDuckGo.
        
        Args:
            query: The search query text
            safe_search_param: Safe search parameter string
            
        Returns:
            Properly encoded URL string for the search
        """
        from urllib.parse import quote
        encoded_query = quote(query)
        return f"https://duckduckgo.com/?q={encoded_query}&atb=v407-1&iar=images&iax=images&ia=images{safe_search_param}"

###############################################################################
#                        ShowFavoritesDialog Class                            #
###############################################################################
class ShowFavoritesDialog(QDialog):
    def show_preview_dialog(self, preview_data):
        """Shows the image preview dialog."""
        logger.debug(f"show_preview_dialog called with preview_data: {preview_data}")
        # Create the dialog
        dialog = ImagePreviewDialog(preview_data, self)
        self.image_manager = image_manager
        self.setWindowTitle("Favorite Wallpapers")
        self.setMinimumSize(400, 300)
        logger.debug(f"show_preview_dialog called with preview_data: {preview_data}")
        dialog = ImagePreviewDialog(preview_data, self)
    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search favorites...")
        self.search_bar.textChanged.connect(self.filter_favorites)
        layout.addWidget(self.search_bar)

        self.fav_list = QListWidget()
        layout.addWidget(self.fav_list)

        btn_layout = QHBoxLayout()
        remove_fav_btn = QPushButton("Remove from Favorites")
        remove_fav_btn.clicked.connect(self.remove_selected_favorite)
        btn_layout.addWidget(remove_fav_btn)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)

        layout.addLayout(btn_layout)

        self.populate_favorites()

    def populate_favorites(self):
        self.fav_list.clear()
        favorites = self.image_manager.get_favorites()
        for path_str, last_used in favorites:
            fname = os.path.basename(path_str)
            tags = self.image_manager.get_tags(path_str)
            tags_str = f" [Tags: {', '.join(tags)}]" if tags else ""
            item_text = fname + tags_str
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, path_str)
            self.fav_list.addItem(item)

    def filter_favorites(self):
        search_text = self.search_bar.text().lower()
        for i in range(self.fav_list.count()):
            item = self.fav_list.item(i)
            path_str = item.data(Qt.UserRole)
            tags = self.image_manager.get_tags(path_str)
            tags_text = " ".join(tags).lower()
            txt = item.text().lower()
            should_show = True
            if search_text:
                should_show = search_text in txt or search_text in tags_text
            item.setHidden(not should_show)

    def remove_selected_favorite(self):
        current_item = self.fav_list.currentItem()
        if current_item:
            path_str = current_item.data(Qt.UserRole)
            self.image_manager.remove_favorite(path_str)
            self.populate_favorites()

###############################################################################
#                        ImagePreviewDialog Class                             #
###############################################################################
class ImagePreviewDialog(QDialog):
    """Dialog for previewing and selecting images before download."""
    
    def __init__(self, image_metadata, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview Selection")
        self.setMinimumSize(800, 600)
        self.image_metadata = image_metadata
        self.selected_images = []
        self.setup_ui()
        # Initialize with all images selected
        self.update_selection_status()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Info and selection count layout
        info_layout = QHBoxLayout()
        
        # Info label
        info_label = QLabel("Select images to download:")
        info_layout.addWidget(info_label)
        
        # Add a spacer to push the selection count to the right
        info_layout.addStretch()
        
        # Selection counter label
        # Selection counter label
        self.selection_counter = QLabel("<b>0 images selected</b>")
        info_layout.addWidget(self.selection_counter)
        layout.addLayout(info_layout)
        
        # Select all/none buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Clear Selections")
        select_none_btn.clicked.connect(self.select_none)
        button_layout.addWidget(select_none_btn)
        
        layout.addLayout(button_layout)
        
        # Scrollable area for image grid
        scroll_area = QWidget()
        self.grid_layout = QVBoxLayout(scroll_area)
        
        # Populate the grid with images
        for idx, metadata in enumerate(self.image_metadata):
            item_widget = self.create_preview_item(metadata, idx)
            self.grid_layout.addWidget(item_widget)
        
        # Add the scroll area to a scrollable container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_area)
        layout.addWidget(scroll)
        
        # Custom dialog buttons
        button_layout = QHBoxLayout()
        
        # Add a spacer to push buttons to the right
        button_layout.addStretch()
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.cancel_button)
        
        # OK button (with dynamic text)
        self.ok_button = QPushButton("Initiate Search")
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
    
    def show_preview_dialog(self, preview_data):
        """Shows the preview dialog with the fetched image data"""
        logger.debug(f"show_preview_dialog called with preview_data: {preview_data}")
        logger.debug("Number of preview items: %d", len(preview_data))
        # Create and show the preview dialog
        preview_dialog = ImagePreviewDialog(preview_data, self)
        result = preview_dialog.exec_()
        
        return result
        
    def create_preview_item(self, metadata, idx):
        """Creates a widget for preview item display with checkbox and thumbnail"""
        logger.debug("Creating preview: filename=%s, url=%s, resolution=%s", 
                    metadata.get("filename"), metadata.get("url"), metadata.get("resolution"))
        
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        
        # Checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(True)  # Default to selected
        checkbox.stateChanged.connect(lambda state, i=idx: self.toggle_selection(i, state))
        item_layout.addWidget(checkbox)
        
        # Thumbnail (if available)
        if 'thumbnail_data' in metadata and metadata['thumbnail_data']:
            pixmap = QPixmap()
            pixmap.loadFromData(metadata['thumbnail_data'])
            thumbnail = QLabel()
            thumbnail.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            item_layout.addWidget(thumbnail)
        else:
            # Placeholder for missing thumbnail
            placeholder = QLabel("No preview")
            placeholder.setFixedSize(100, 100)
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #444; color: #aaa; border: 1px solid #555;")
            item_layout.addWidget(placeholder)
        
        # Image info
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        
        # Format image information
        name_label = QLabel(f"<b>Filename:</b> {metadata.get('filename', 'Unknown')}")
        info_layout.addWidget(name_label)
        
        size_label = QLabel(f"<b>Size:</b> {metadata.get('size_kb', 'Unknown')} KB")
        info_layout.addWidget(size_label)
        
        resolution = metadata.get('resolution', 'Unknown')
        if isinstance(resolution, tuple):
            resolution = f"{resolution[0]}x{resolution[1]}"
        resolution_label = QLabel(f"<b>Resolution:</b> {resolution}")
        info_layout.addWidget(resolution_label)
        
        source_label = QLabel(f"<b>Source:</b> {metadata.get('source', 'Unknown')}")
        info_layout.addWidget(source_label)
        
        item_layout.addWidget(info_widget)
        item_layout.addStretch()
        
        # Add this item to selected by default
        self.selected_images.append(idx)
        return item_widget
    
    def toggle_selection(self, idx, state):
        if state == Qt.Checked and idx not in self.selected_images:
            self.selected_images.append(idx)
        elif state == Qt.Unchecked and idx in self.selected_images:
            self.selected_images.remove(idx)
        self.update_selection_status()
        
    def update_selection_status(self):
        """Update the OK button text and selection counter based on selection state"""
        count = len(self.selected_images)
        
        # Update the selection counter with bold formatting
        self.selection_counter.setText(f"<b>{count} image{'s' if count != 1 else ''} selected</b>")
        
        # Update the OK button text and enabled state
        if count > 0:
            self.ok_button.setText("Start Download")
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setText("Initiate Search")
            self.ok_button.setEnabled(False)
    
    def select_all(self):
        """Select all images in the grid"""
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                # Find the checkbox in the widget's layout
                for j in range(widget.layout().count()):
                    item = widget.layout().itemAt(j).widget()
                    if isinstance(item, QCheckBox):
                        item.setChecked(True)
        self.update_selection_status()
        
    def select_none(self):
        """Clear all image selections"""
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                # Find the checkbox in the widget's layout
                for j in range(widget.layout().count()):
                    item = widget.layout().itemAt(j).widget()
                    if isinstance(item, QCheckBox):
                        item.setChecked(False)
        self.update_selection_status()
    def get_selected_metadata(self):
        """Return the metadata for selected images only."""
        return [self.image_metadata[idx] for idx in self.selected_images]

###############################################################################
#                           MainWindow Class                                  #
###############################################################################
class MainWindow(QMainWindow):
    """
    Main application window for Enhanced Wallpaper Manager.
    
    This class implements the main GUI for the application, handling all user
    interactions, configuration, and coordinating between different components
    like the ImageManager and worker threads. It provides functionality for
    browsing, managing, and setting wallpapers, as well as downloading new
    wallpapers from various online sources.
    
    The GUI is organized into multiple tabs (Wallpaper, Download, Settings)
    and provides both a file selector and preview pane for viewing wallpapers.
    
    Attributes:
        config (dict): Application configuration dictionary
        image_manager (ImageManager): Manager for wallpaper images
        wallpapers_data (list): List of wallpaper data for display
        wallpaper_worker (WallpaperWorker): Thread for automatic wallpaper changing
        scraper_worker (MultiSourceScraperWorker): Thread for downloading wallpapers
        refresh_timer (QTimer): Timer for refreshing the wallpaper list
        mutex (QMutex): Mutex for thread-safe operations
        current_view (str): Current view mode ("all" or "favorites")
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Wallpaper Manager")
        self.setMinimumSize(800, 600)
        # Initialize critical components first
        self.mutex = QMutex()
        self.views_stack = None
        self.icon_list = None
        self.list_list = None
        self.details_table = None

        # 1) Load config
        self.config = self.load_config()

        # 2) Create image manager with error handling
        try:
            self.image_manager = ImageManager(self.config['wallpaper_directory'])
        except Exception as e:
            logger.error(f"Failed to initialize ImageManager: {e}")
            self.show_error(f"Failed to initialize ImageManager: {e}")

        # 3) Prepare data
        self.wallpapers_data = []

        # 4) Proxy model for future sorting/filtering
        self.proxy_model = QSortFilterProxyModel(self)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

        # 5) Setup UI
        self.setup_ui()
        self.setup_tray()

        # 6) Workers
        self.wallpaper_worker = None
        self.scraper_worker = None

        # 7) Refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_wallpaper_data)
        self.update_refresh_timer()

        # 8) Populate data
        # Call refresh_wallpaper_data to populate grid view on startup
        self.refresh_wallpaper_data()
        
        # 9) Possibly autostart
        if self.config.get('autostart', False):
            self.start_wallpaper_changer()
    ########################################################################
    #                           Config Methods                              #
    ########################################################################
    #                           UI Setup Methods                            #
    ########################################################################
    # This is now removed since we've fixed the implementation at line 1906
    ########################################################################
    ########################################################################
    #                           Config Methods                              #
    ########################################################################
    def load_config(self):
        """
        Load configuration from a JSON file or create a default configuration if the file doesn't exist.
        
        This method reads configuration settings from a YAML file in the user's home directory.
        If the file doesn't exist, it uses default values for all settings. The configuration
        includes settings for wallpaper directory, change intervals, scraper settings, and more.
        
        Returns:
            dict: A dictionary containing all configuration settings with their values
        """
        # Define default configuration values
        default_config = {
            'wallpaper_directory': Path.home() / 'Pictures' / 'Wallpapers',
            'interval': 2400,
            'autostart': False,
            'minimize_to_tray_on_start': True,
            'show_notifications': True,
            'notification_duration': 3,
            'display_all_before_repeat': False,
            'active_hours_start': 9,
            'active_hours_end': 17,
            'scraper_category': 'landscapes',
            'scraper_resolution': 'Any',
            'scraper_limit': 10,
            'max_wallpapers': 100,
            'enabled_sources': ['wallhaven', 'duckduckgo'],
            'optimize_images': True,
            'detect_duplicates': True,
            'check_quality': True,
            'preview_refresh_interval': 60,
            'tray_icon_path': os.path.join(os.path.dirname(__file__), "wallpaper-menu-bar-icon.png"),
            'search_keywords': ['nature', 'abstract', 'landscape', 'space'],
            'max_download_count': 10,
            'purity_filter': 110,  # Default to SFW + Sketchy
            'change_on_start': True,  # Default to changing wallpaper on start
            'custom_keyword': '',  # Default empty string for custom keyword
        }
        
        # Define the configuration file path
        config_path = Path.home() / '.wallpaper_changer.yaml'
        
        try:
            # Try to load existing configuration
            logger.debug(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Merge default config with loaded data (default values will be used for any missing keys)
            config = {**default_config, **(data or {})}
            logger.debug("Configuration loaded successfully")
            
            # Ensure wallpaper directory exists
            wallpaper_dir = Path(config['wallpaper_directory'])
            if not wallpaper_dir.exists():
                logger.info(f"Creating wallpaper directory: {wallpaper_dir}")
                wallpaper_dir.mkdir(parents=True, exist_ok=True)
                
            return config
            
        except FileNotFoundError:
            # Configuration file doesn't exist, create directory and use defaults
            logger.info(f"Configuration file not found. Using default configuration.")
            
            # Ensure wallpaper directory exists
            wallpaper_dir = Path(default_config['wallpaper_directory'])
            if not wallpaper_dir.exists():
                logger.info(f"Creating wallpaper directory: {wallpaper_dir}")
                wallpaper_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the default configuration
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(default_config, f)
                logger.info(f"Default configuration saved to {config_path}")
            except Exception as e:
                logger.error(f"Error saving default configuration: {str(e)}")
                
            return default_config
        except Exception as e:
            # Handle other potential errors
            logger.error(f"Error loading configuration: {str(e)}")
            return default_config
    # _thread_safe method is now defined in the MainWindow class
    
    @contextmanager
    def _thread_safe(self):
        """Context manager for thread-safe operations."""
        try:
            self.mutex.lock()
            yield
        finally:
            self.mutex.unlock()
            
    def save_config(self):
        """
        Save configuration to a YAML file in the user's home directory.
        
        This method collects current settings from the UI components and saves them to a YAML file.
        If UI components are not available (e.g., during shutdown), it falls back to current
        configuration values. The method uses a thread-safe context manager to ensure safe
        file operations in a multi-threaded environment.
        
        Returns:
            bool: True if successful, False if an error occurred
        """
        # Get current config to use as fallback values
        current_config = self.config.copy() if hasattr(self, 'config') else {}
        
        # Get wallpaper directory
        try:
            wallpaper_dir = self.dir_label.text()
        except (RuntimeError, AttributeError):
            wallpaper_dir = current_config.get('wallpaper_directory', '')
            
        # Get custom keyword
        try:
            custom_keyword = self.custom_keyword_input.text()
        except (RuntimeError, AttributeError):
            custom_keyword = current_config.get('custom_keyword', '')
            
        # Get interval
        try:
            interval = self.interval_spin.value() * 60
        except (RuntimeError, AttributeError):
            interval = current_config.get('interval', 900)  # Default 15 minutes
            
        # Get autostart
        try:
            autostart = self.autostart_cb.isChecked()
        except (RuntimeError, AttributeError):
            autostart = current_config.get('autostart', False)
            
        # Get minimize to tray
        try:
            minimize_to_tray = self.minimize_cb.isChecked()
        except (RuntimeError, AttributeError):
            minimize_to_tray = current_config.get('minimize_to_tray_on_start', True)
            
        # Get sources
        enabled_sources = []
        try:
            for source, checkbox in self.source_checkboxes.items():
                try:
                    if checkbox.isChecked():
                        enabled_sources.append(source)
                except (RuntimeError, AttributeError):
                    pass  # Skip if checkbox is deleted
            
            # Make sure at least one source is enabled
            if not enabled_sources and 'enabled_sources' in current_config:
                enabled_sources = current_config['enabled_sources']
            if not enabled_sources:
                enabled_sources = ['wallhaven']
        except (RuntimeError, AttributeError):
            enabled_sources = current_config.get('enabled_sources', ['wallhaven'])
            
        # Get preview refresh interval
        try:
            preview_refresh_interval = self.preview_refresh_spin.value()
        except (RuntimeError, AttributeError):
            preview_refresh_interval = current_config.get('preview_refresh_interval', 60)
            
        # Get purity filter
        try:
            purity_filter = self.get_purity_filter_value()
        except (RuntimeError, AttributeError):
            purity_filter = current_config.get('purity_filter', 110)  # Default to SFW + Sketchy
            
        # Get change on start setting
        try:
            change_on_start = self.change_on_start_cb.isChecked()
        except (RuntimeError, AttributeError):
            change_on_start = current_config.get('change_on_start', True)

        # Get scraper settings
        try:
            scraper_category = self.category_combo.currentText()
        except (RuntimeError, AttributeError):
            scraper_category = current_config.get('scraper_category', 'landscapes')

        try:
            scraper_resolution = self.resolution_combo.currentText()
        except (RuntimeError, AttributeError):
            scraper_resolution = current_config.get('scraper_resolution', 'Any')

        try:
            scraper_limit = self.download_limit_spin.value()
        except (RuntimeError, AttributeError):
            scraper_limit = current_config.get('scraper_limit', 10)

        try:
            max_wallpapers = self.max_wallpapers_spin.value()
        except (RuntimeError, AttributeError):
            max_wallpapers = current_config.get('max_wallpapers', 100)
            
        try:
            display_all = self.display_all_cb.isChecked()
        except (RuntimeError, AttributeError):
            display_all = current_config.get('display_all_before_repeat', False)

        # Get notification settings
        try:
            show_notifications = self.notify_cb.isChecked()
        except (RuntimeError, AttributeError):
            show_notifications = current_config.get('show_notifications', True)
        try:
            notify_duration = self.duration_spin.value()
        except (RuntimeError, AttributeError):
            notify_duration = current_config.get('notification_duration', 3)
            
        # Get active hours settings
        try:
            active_hours_start = self.start_time.time().hour()
            active_hours_end = self.end_time.time().hour()
        except (RuntimeError, AttributeError):
            active_hours_start = current_config.get('active_hours_start', 9)
            active_hours_end = current_config.get('active_hours_end', 17)
            
        # Build the updated config dictionary
        updated_config = {
            'wallpaper_directory': str(wallpaper_dir),
            'interval': interval,
            'autostart': autostart,
            'minimize_to_tray_on_start': minimize_to_tray,
            'show_notifications': show_notifications,
            'notification_duration': notify_duration,
            'display_all_before_repeat': display_all,
            'active_hours_start': active_hours_start,
            'active_hours_end': active_hours_end,
            'scraper_category': scraper_category,
            'scraper_resolution': scraper_resolution,
            'scraper_limit': scraper_limit,
            'max_wallpapers': max_wallpapers,
            'enabled_sources': enabled_sources,
            'preview_refresh_interval': preview_refresh_interval,
            'purity_filter': purity_filter,
            'change_on_start': change_on_start,
            'custom_keyword': custom_keyword,
        }
        
        # Preserve any other keys that might be in the current config but not updated here
        for key, value in current_config.items():
            if key not in updated_config:
                updated_config[key] = value
                
        # Update the instance config
        self.config.update(updated_config)
        
        # Save the config to file using thread-safe operations
        with self._thread_safe():
            try:
                # Define the configuration file path
                config_path = Path.home() / '.wallpaper_changer.yaml'
                
                logger.debug(f"Saving configuration to {config_path}")
                
                # Write to the file
                with open(config_path, 'w') as f:
                    yaml.dump(updated_config, f)
                    
                logger.debug("Configuration saved successfully")
                self.log_message("Settings saved successfully")
                return True
                
            except Exception as e:
                error_msg = f"Error saving configuration: {str(e)}"
                logger.error(error_msg)
                self.show_error(error_msg)
                return False

    def get_purity_filter_value(self):
        """Convert the purity filter dropdown selection to the corresponding API value."""
        index = self.purity_combo.currentIndex()
        if index == 0:
            return 100  # SFW Only
        elif index == 1:
            return 110  # SFW + Sketchy
        else:
            return 111  # All

    def update_refresh_timer(self):
        minutes = self.config.get('preview_refresh_interval', 60)
        self.refresh_timer.stop()
        self.refresh_timer.start(minutes * 60 * 1000)

    ########################################################################
    #                             UI Setup                                  #
    ########################################################################
    def setup_ui(self):
        """
        Initialize the main UI components and layout for the application.
        
        This method sets up the central widget, tab structure, splitters for the
        wallpaper browser, file selector pane, preview pane, and status bar 
        with progress indicator.
        """
        logger.debug("Setting up main UI")
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create file and preview splitter
        self.file_and_preview_splitter = QSplitter(Qt.Horizontal)
        
        # Set up file selector pane
        file_selector = self.create_file_selector_pane()
        self.file_and_preview_splitter.addWidget(file_selector)
        
        # Set up preview pane
        preview_pane = self.create_preview_pane()
        self.file_and_preview_splitter.addWidget(preview_pane)
        
        # Set initial sizes for the splitter
        self.file_and_preview_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        # Create tabs for different functionality
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_wallpaper_tab(), "Wallpaper")
        self.tabs.addTab(self.create_scraper_tab(), "Download")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        
        # Add splitter and tabs to main layout
        main_layout.addWidget(self.file_and_preview_splitter, 2)  # Give it more stretch
        main_layout.addWidget(self.tabs, 1)
        
        # Set up status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Add progress bar to status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.setMaximumWidth(200)
        self.statusBar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()  # Hide initially until needed
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Set initial status message
        self.statusBar.showMessage("Ready")
    def create_wallpaper_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        dir_group = QGroupBox("Wallpaper Directory")
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel(self.config['wallpaper_directory'])
        dir_layout.addWidget(self.dir_label)

        choose_dir_btn = QPushButton("Choose Directory")
        choose_dir_btn.clicked.connect(self.choose_directory)
        dir_layout.addWidget(choose_dir_btn)

        show_folder_btn = QPushButton("Show in Folder")
        show_folder_btn.clicked.connect(self.open_wallpaper_folder)
        dir_layout.addWidget(show_folder_btn)


        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout()

        self.display_all_cb = QCheckBox("Display all wallpapers before repeating")
        self.display_all_cb.setChecked(self.config.get('display_all_before_repeat', False))
        display_layout.addWidget(self.display_all_cb)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Change interval (minutes):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 1440)
        self.interval_spin.setValue(self.config['interval'] // 60)
        interval_layout.addWidget(self.interval_spin)
        display_layout.addLayout(interval_layout)

        refresh_layout = QHBoxLayout()
        refresh_layout.addWidget(QLabel("Refresh frequency (minutes):"))
        self.preview_refresh_spin = QSpinBox()
        self.preview_refresh_spin.setRange(1, 1440)
        self.preview_refresh_spin.setValue(self.config.get('preview_refresh_interval', 60))
        refresh_layout.addWidget(self.preview_refresh_spin)
        display_layout.addLayout(refresh_layout)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        maintenance_group = QGroupBox("Maintenance")
        maint_layout = QVBoxLayout()
        self.clear_wallpapers_btn = QPushButton("Remove Old or Low-Quality Wallpapers")
        self.clear_wallpapers_btn.clicked.connect(self.cleanup_directory)
        maint_layout.addWidget(self.clear_wallpapers_btn)

        stats_label = QLabel("Wallpaper Statistics:")
        maint_layout.addWidget(stats_label)

        self.stats_list = QListWidget()
        maint_layout.addWidget(self.stats_list)

        maintenance_group.setLayout(maint_layout)
        layout.addWidget(maintenance_group)

        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_wallpaper_changer)
        controls_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_wallpaper_changer)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)

        next_btn = QPushButton("Next Wallpaper")
        next_btn.clicked.connect(self.next_wallpaper)
        controls_layout.addWidget(next_btn)

        controls_layout.addStretch()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_config)
        controls_layout.addWidget(save_btn)

        hide_btn = QPushButton("Hide")
        hide_btn.clicked.connect(self.hide)
        controls_layout.addWidget(hide_btn)

        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(QApplication.quit)
        controls_layout.addWidget(quit_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        self.log_list = QListWidget()
        log_layout.addWidget(self.log_list)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        widget.setLayout(layout)
        return widget


    def create_scraper_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Add a header layout with gear button in top-right
        header_layout = QHBoxLayout()
        header_layout.addStretch(1)  # Push gear button to the right
        
        # Create settings gear button
        settings_button = QPushButton()
        settings_button.setIcon(QIcon.fromTheme("preferences-system", QIcon(":/icons/settings.png")))
        settings_button.setToolTip("Open Settings")
        settings_button.setMaximumSize(32, 32)
        settings_button.setFlat(True)
        settings_button.clicked.connect(lambda: self.tabs.setCurrentIndex(2))  # Switch to settings tab (index 2)
        header_layout.addWidget(settings_button)
        
        layout.addLayout(header_layout)

        sources_group = QGroupBox("Wallpaper Sources")
        sources_layout = QVBoxLayout()
        self.source_checkboxes = {}
        for source in ['wallhaven', 'duckduckgo']:
            cb = QCheckBox(source.capitalize())
            cb.setChecked(source in self.config.get('enabled_sources', ['wallhaven']))
            self.source_checkboxes[source] = cb
            sources_layout.addWidget(cb)

        sources_group.setLayout(sources_layout)
        layout.addWidget(sources_group)

        settings_group = QGroupBox("Download Settings")
        settings_layout = QVBoxLayout()

        # Custom keyword input field
        custom_keyword_layout = QHBoxLayout()
        custom_keyword_layout.addWidget(QLabel("Custom Keyword:"))
        self.custom_keyword_input = QLineEdit()
        self.custom_keyword_input.setPlaceholderText("Enter custom keywords to search...")
        self.custom_keyword_input.setText(self.config.get('custom_keyword', ''))
        custom_keyword_layout.addWidget(self.custom_keyword_input)
        settings_layout.addLayout(custom_keyword_layout)

        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(['landscapes', 'nature', 'abstract', 'space', 'art', 'beautiful women', 'nude women'])
        self.category_combo.setCurrentText(self.config.get('scraper_category', 'landscapes'))
        cat_layout.addWidget(self.category_combo)
        settings_layout.addLayout(cat_layout)

        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(['Any', '1920x1080', '2560x1440', '3840x2160'])
        self.resolution_combo.setCurrentText(self.config.get('scraper_resolution', 'Any'))
        res_layout.addWidget(self.resolution_combo)
        settings_layout.addLayout(res_layout)
        # Add purity filter dropdown
        purity_layout = QHBoxLayout()
        purity_layout.addWidget(QLabel("Purity Filter:"))
        self.purity_combo = QComboBox()
        self.purity_combo.addItems(['SFW Only', 'SFW + Sketchy', 'All'])
        # Set the current selection based on config
        purity_value = self.config.get('purity_filter', 110)
        if purity_value == 100:
            self.purity_combo.setCurrentIndex(0)
        elif purity_value == 110:
            self.purity_combo.setCurrentIndex(1)
        else:  # 111
            self.purity_combo.setCurrentIndex(2)
        purity_layout.addWidget(self.purity_combo)
        settings_layout.addLayout(purity_layout)

        limit_layout = QHBoxLayout()
        limit_label = QLabel("Max Local Wallpapers to Keep:")
        limit_label.setToolTip("The maximum number of wallpapers stored locally.")
        limit_layout.addWidget(limit_label)
        self.max_wallpapers_spin = QSpinBox()
        self.max_wallpapers_spin.setRange(1, 10000)
        self.max_wallpapers_spin.setValue(self.config.get('max_wallpapers', 100))
        limit_layout.addWidget(self.max_wallpapers_spin)
        settings_layout.addLayout(limit_layout)

        download_layout = QHBoxLayout()
        download_layout.addWidget(QLabel("Number to Download (scraper limit):"))
        self.download_limit_spin = QSpinBox()
        self.download_limit_spin.setRange(1, 9999)
        self.download_limit_spin.setValue(self.config.get('scraper_limit', 10))
        download_layout.addWidget(self.download_limit_spin)
        settings_layout.addLayout(download_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()

        scrape_btn = QPushButton("Start Download")
        scrape_btn.clicked.connect(self.start_scraping)
        controls_layout.addWidget(scrape_btn)

        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.clicked.connect(self.save_config)
        controls_layout.addWidget(save_settings_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        log_group = QGroupBox("Download Log")
        log_layout = QVBoxLayout()
        self.scraper_log = QListWidget()
        log_layout.addWidget(self.scraper_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        widget.setLayout(layout)
        return widget

    def create_settings_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        startup_group = QGroupBox("Startup Settings")
        startup_layout = QVBoxLayout(startup_group)

        self.autostart_cb = QCheckBox("Start automatically at login")
        self.autostart_cb.setChecked(self.config.get('autostart', False))
        startup_layout.addWidget(self.autostart_cb)

        self.minimize_cb = QCheckBox("Start minimized to menu bar")
        self.minimize_cb.setChecked(self.config.get('minimize_to_tray_on_start', True))
        startup_layout.addWidget(self.minimize_cb)
        
        self.change_on_start_cb = QCheckBox("Change wallpaper on start")
        self.change_on_start_cb.setChecked(self.config.get('change_on_start', True))
        startup_layout.addWidget(self.change_on_start_cb)

        startup_group.setLayout(startup_layout)
        layout.addWidget(startup_group)

        hours_group = QGroupBox("Active Hours")
        hours_layout = QHBoxLayout(hours_group)  # This assigns the layout to hours_group

        start_label = QLabel("Start Time:")
        self.start_time = QTimeEdit()
        self.start_time.setTime(QDateTime.fromTime_t(self.config['active_hours_start'] * 3600).time())

        end_label = QLabel("End Time:")
        self.end_time = QTimeEdit()
        self.end_time.setTime(QDateTime.fromTime_t(self.config['active_hours_end'] * 3600).time())

        hours_layout.addWidget(start_label)
        hours_layout.addWidget(self.start_time)
        hours_layout.addWidget(end_label)
        hours_layout.addWidget(self.end_time)
        layout.addWidget(hours_group)

        notify_group = QGroupBox("Notifications")
        notify_layout = QVBoxLayout()

        self.notify_cb = QCheckBox("Show notifications")
        self.notify_cb.setChecked(self.config.get('show_notifications', True))
        notify_layout.addWidget(self.notify_cb)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Notification duration (seconds):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 10)
        self.duration_spin.setValue(self.config.get('notification_duration', 3))
        duration_layout.addWidget(self.duration_spin)
        notify_layout.addLayout(duration_layout)

        notify_group.setLayout(notify_layout)
        layout.addWidget(notify_group)

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    ########################################################################
    #                       File Selector Pane Methods                      #
    ########################################################################
    def create_file_selector_pane(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Search row
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search by filename, tag name...")
        self.search_bar.textChanged.connect(self.filter_wallpapers)
        search_layout.addWidget(self.search_bar)

        self.show_favorites_btn = QPushButton("Show Favorites")
        self.show_favorites_btn.setCheckable(True)
        self.show_favorites_btn.clicked.connect(self.toggle_favorites_view)
        search_layout.addWidget(self.show_favorites_btn)

        layout.addLayout(search_layout)

        # View controls
        view_controls = QHBoxLayout()
        self.preview_view_selector = QComboBox()
        self.preview_view_selector.addItems(["Icon", "List", "Details"])
        self.preview_view_selector.currentIndexChanged.connect(self.switch_view_mode)
        view_controls.addWidget(self.preview_view_selector)

        self.layout_button_horizontal = QToolButton()
        # Use standard icon if image file doesn't exist
        if Path("horizontal.png").exists():
            self.layout_button_horizontal.setIcon(QIcon("horizontal.png"))
        else:
            self.layout_button_horizontal.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.layout_button_horizontal.clicked.connect(self.set_horizontal_layout)
        view_controls.addWidget(self.layout_button_horizontal)

        self.layout_button_vertical = QToolButton()
        # Use standard icon if image file doesn't exist
        if Path("vertical.png").exists():
            self.layout_button_vertical.setIcon(QIcon("vertical.png"))
        else:
            self.layout_button_vertical.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        self.layout_button_vertical.clicked.connect(self.set_vertical_layout)
        view_controls.addWidget(self.layout_button_vertical)

        layout.addLayout(view_controls)

        # Create the views_stack
        self.views_stack = QStackedWidget()
        self.views_stack.setObjectName("WallpaperViews")  # Add unique object name for easier debugging

        # Icon view
        self.icon_list = QListWidget()
        self.icon_list.setViewMode(QListView.IconMode)
        self.icon_list.setIconSize(QSize(100, 100))
        self.icon_list.setMovement(QListView.Static)
        self.icon_list.setResizeMode(QListView.Adjust)
        self.icon_list.setSpacing(10)  # Add spacing between icons
        self.icon_list.setUniformItemSizes(True)  # Helps with performance
        self.icon_list.itemSelectionChanged.connect(self.icon_selection_changed)
        self.icon_list.setFocusPolicy(Qt.StrongFocus)
        self.icon_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        # List view
        self.list_list = QListWidget()
        self.list_list.setViewMode(QListView.ListMode)
        self.list_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_list.itemSelectionChanged.connect(self.list_selection_changed)
        self.list_list.setFocusPolicy(Qt.StrongFocus)
        self.list_list.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Details view
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(6)
        self.details_table.setHorizontalHeaderLabels(["Name","Size (KB)","Date Modified","Kind","Fave","Tags"])
        self.details_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.details_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.details_table.horizontalHeader().setStretchLastSection(True)
        self.details_table.horizontalHeader().setSortIndicatorShown(True)
        self.details_table.setSortingEnabled(True)
        self.details_table.itemSelectionChanged.connect(self.details_selection_changed)
        self.details_table.setFocusPolicy(Qt.StrongFocus)
        self.details_table.cellDoubleClicked.connect(self.on_table_cell_double_clicked)

        # Add views to the stack
        self.views_stack.addWidget(self.icon_list)
        self.views_stack.addWidget(self.list_list)
        self.views_stack.addWidget(self.details_table)
        
        # Set default view based on config (fixing duplicate line)
        default_view = self.config.get('default_view', 'icon')
        if default_view == 'list':
            self.views_stack.setCurrentIndex(1)
            self.preview_view_selector.setCurrentIndex(1)
        elif default_view == 'details':
            self.views_stack.setCurrentIndex(2)
            self.preview_view_selector.setCurrentIndex(2)
        else:
            # Default to icon view
            self.views_stack.setCurrentIndex(0)
            self.preview_view_selector.setCurrentIndex(0)
            self.views_stack.setCurrentIndex(0)
            self.preview_view_selector.setCurrentIndex(0)
        
        # Add the views_stack to the layout
        layout.addWidget(self.views_stack)
        
        # Initial data load - trigger data load if not already loaded
        if not self.wallpapers_data:
            QTimer.singleShot(100, self.refresh_wallpaper_data)
            
        # Return the created widget
        return widget
    def switch_view_mode(self, index: int):
        """Switch among Icon/List/Details."""
        self.views_stack.setCurrentIndex(index)
        
    def keyPressEvent(self, event):
        """Handle key press events for the application."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # If Enter/Return key is pressed, set the currently selected wallpaper
            if self.get_current_selected_path():
                self.assign_preview_wallpaper()
        else:
            super().keyPressEvent(event)
            
    def on_item_double_clicked(self, item):
        """Handle double-click on list or icon view items."""
        if item and item.data(Qt.UserRole):
            self.assign_preview_wallpaper()

    def on_table_cell_double_clicked(self, row, column):
        """Handle double-click on table cells in details view."""
        item = self.details_table.item(row, 0)  # Get the item in the first column (name)
        if item and item.data(Qt.UserRole):
            self.assign_preview_wallpaper()

    # This is a duplicate method - removing it as we're using the complete implementation at line 2207
    def create_preview_pane(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.large_preview_label = QLabel("Image Preview")
        self.large_preview_label.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.large_preview_label.setAlignment(Qt.AlignCenter)
        self.large_preview_label.setScaledContents(True)
        layout.addWidget(self.large_preview_label, stretch=1)

        # Tag input
        tag_layout = QHBoxLayout()
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Add tags (comma-separated)")
        tag_layout.addWidget(self.tag_input)

        add_tag_btn = QPushButton("Add Tags")
        add_tag_btn.clicked.connect(self.add_tags_to_current)
        tag_layout.addWidget(add_tag_btn)
        layout.addLayout(tag_layout)
        # Layout for explanation text
        note_layout = QHBoxLayout()
        note_label = QLabel("Double-click an image or press Enter to set as wallpaper")
        note_label.setAlignment(Qt.AlignCenter)
        note_layout.addWidget(note_label)
        layout.addLayout(note_layout)

        # Favorite controls
        fav_layout = QHBoxLayout()
        self.add_favorite_btn = QPushButton("Add Favorite")
        self.add_favorite_btn.setEnabled(False)
        self.add_favorite_btn.clicked.connect(self.add_favorite)
        fav_layout.addWidget(self.add_favorite_btn)

        self.remove_favorite_btn = QPushButton("Remove Favorite")
        self.remove_favorite_btn.setEnabled(False)
        self.remove_favorite_btn.clicked.connect(self.remove_favorite)
        fav_layout.addWidget(self.remove_favorite_btn)

        layout.addLayout(fav_layout)

        return widget

    ########################################################################
    #                       Wallpaper Setting Methods                       #
    ########################################################################
    def assign_preview_wallpaper(self):
        """
        Actually sets the selected wallpaper as the system wallpaper.
        """
        path = self.get_current_selected_path()
        if not path:
            self.show_error("No wallpaper selected.")
            return
        try:
            if platform.system() == 'Darwin':
                script = f'''
                tell application "System Events"
                    tell every desktop
                        set picture to "{path}"
                    end tell
                end tell
                '''
                subprocess.run(["osascript", "-e", script], check=True)
            elif platform.system() == 'Linux':
                # Ubuntu Budgie uses gsettings for changing wallpaper
                try:
                    # Check for Ubuntu Budgie
                    desktop_session = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
                    if 'budgie' in desktop_session:
                        # For Budgie desktop
                        subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri', f"file://{path}"], check=True)
                        # Also set the dark mode variant just in case
                        subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri-dark', f"file://{path}"], check=True)
                    else:
                        # Generic method for other Linux environments
                        desktop_env = os.environ.get('XDG_CURRENT_DESKTOP', '').lower()
                        if 'gnome' in desktop_env:
                            subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri', f"file://{path}"], check=True)
                        elif 'kde' in desktop_env:
                            script = f"""
                            var allDesktops = desktops();
                            for (var i=0; i<allDesktops.length; i++) {{
                                d = allDesktops[i];
                                d.wallpaperPlugin = "org.kde.image";
                                d.currentConfigGroup = Array("Wallpaper", "org.kde.image", "General");
                                d.writeConfig("Image", "{path}");
                            }}
                            """
                            subprocess.run(['qdbus', 'org.kde.plasmashell', '/PlasmaShell', 'evaluateScript', script], check=True)
                        elif 'xfce' in desktop_env:
                            subprocess.run(['xfconf-query', '-c', 'xfce4-desktop', '-p', '/backdrop/screen0/monitor0/workspace0/last-image', '-s', path], check=True)
                        elif 'mate' in desktop_env:
                            subprocess.run(['gsettings', 'set', 'org.mate.background', 'picture-filename', path], check=True)
                        else:
                            # Try a more generic method for other desktops
                            subprocess.run(['gsettings', 'set', 'org.gnome.desktop.background', 'picture-uri', f"file://{path}"], check=True)
                except Exception as e:
                    self.show_error(f"Error setting wallpaper on Linux: {e}")
            elif platform.system() == 'Windows':
                import ctypes
                SPI_SETDESKWALLPAPER = 20
                ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, path, 3)
        except subprocess.CalledProcessError as e:
            self.show_error(f"Error assigning wallpaper: {e}")

    def assign_preview_wallpaper_and_pause(self):
        """
        Assigns the wallpaper, then stops the wallpaper worker if running.
        """
        self.assign_preview_wallpaper()
        if self.wallpaper_worker:
            self.wallpaper_worker.stop()
            self.log_message("Wallpaper changer paused.")

    ########################################################################
    #                       Data Handling & Filtering                       #
    ########################################################################
    def refresh_wallpaper_data(self):
        directory = self.config['wallpaper_directory']
        if not os.path.exists(directory):
            logger.error(f"Wallpaper directory does not exist: {directory}")
            self.show_error(f"Wallpaper directory does not exist: {directory}")
            return

        # Ensure views are initialized
        if not hasattr(self, 'icon_list') or not hasattr(self, 'list_list') or not hasattr(self, 'details_table'):
            self.show_error("Grid view components not initialized")
            return

        # Move the heavy operations to a background thread
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

        try:
            self.scan_worker = ScanWorker(directory, valid_extensions, self.image_manager)
            self.scan_worker.data_ready.connect(self.update_wallpaper_data)
            self.scan_worker.start()
            logger.debug("Started scan worker")
        except Exception as e:
            logger.error(f"Failed to start scan worker: {e}")
            self.show_error(f"Failed to start scan: {e}")

    def update_wallpaper_data(self, data):
        """Called when scan worker completes"""
        self.wallpapers_data = data
        self.filter_wallpapers()
    def filter_wallpapers(self, search_text=None):
        # Get search text from search bar if not provided
        if search_text is None and hasattr(self, 'search_bar'):
            search_text = self.search_bar.text().lower()
        else:
            search_text = search_text or ""
            
        # Check if show_favorites_btn exists before trying to access it
        show_favorites = self.show_favorites_btn.isChecked() if hasattr(self, 'show_favorites_btn') else False
        
        filtered_data = []
        for item in self.wallpapers_data:
            if show_favorites and not item["favorite"]:
                continue
            if search_text:
                name_match = (search_text in item["name"].lower())
                tags_text = " ".join(item["tags"]).lower()
                tags_match = (search_text in tags_text)
                if not (name_match or tags_match):
                    continue
            filtered_data.append(item)

        self.populate_views_with_data(filtered_data)
        
    def populate_views_with_data(self, data):
        """
        Populate all view components with the provided data.
        
        This method updates the icon view, list view, and details view with the
        given data, handling errors gracefully if any of the views aren't properly
        initialized. It also maintains the current selection if possible.
        
        Args:
            data: List of dictionaries containing wallpaper data
        """
        # Ensure views are initialized
        if not hasattr(self, 'icon_list') or not hasattr(self, 'list_list') or not hasattr(self, 'details_table'):
            logger.error("Grid view components not initialized")
            self.show_error("Grid view not initialized")
            return
        
        # Check which views are available and not None
        has_icon_list = hasattr(self, 'icon_list') and self.icon_list is not None
        has_list_list = hasattr(self, 'list_list') and self.list_list is not None
        has_details_table = hasattr(self, 'details_table') and self.details_table is not None
        
        # Clear existing data with progress feedback
        if data:
            logger.debug(f"Populating views with {len(data)} items")
            if has_icon_list:
                self.icon_list.clear()
        if has_list_list:
            self.list_list.clear()
        if has_details_table:
            self.details_table.setRowCount(0)

        # Icon view - only populate if the attribute exists
        if has_icon_list:
            for item in data:
                witem = QListWidgetItem(item["name"])
                witem.setData(Qt.UserRole, item["path"])
                try:
                    pix = QPixmap(item["path"])
                    if not pix.isNull():
                        icon = QIcon(pix.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        witem.setIcon(icon)
                    else:
                        # Handle missing or corrupt image
                        witem.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
                except Exception as e:
                    logger.error(f"Error loading image thumbnail for {item['path']}: {e}")
                    witem.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
                self.icon_list.addItem(witem)
        # List view - only populate if the attribute exists
        if has_list_list:
            for item in data:
                witem = QListWidgetItem(item["name"])
                witem.setData(Qt.UserRole, item["path"])
                self.list_list.addItem(witem)
        # Details view - only populate if the attribute exists
        if has_details_table:
            self.details_table.setSortingEnabled(False)
            self.details_table.setRowCount(len(data))
            for row, item in enumerate(data):
                name_item = QTableWidgetItem(item["name"])
                name_item.setData(Qt.UserRole, item["path"])

                size_item = QTableWidgetItem()
                size_item.setData(Qt.DisplayRole, item["size_kb"])
                size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                date_str = item["modified"].strftime("%Y-%m-%d %H:%M:%S")
                date_item = QTableWidgetItem(date_str)

                kind_item = QTableWidgetItem(item["kind"])

                fave_item = QTableWidgetItem("●" if item["favorite"] else "")
                fave_item.setTextAlignment(Qt.AlignCenter)

                tags_item = QTableWidgetItem(", ".join(item["tags"]))

                self.details_table.setItem(row, 0, name_item)
                self.details_table.setItem(row, 1, size_item)
                self.details_table.setItem(row, 2, date_item)
                self.details_table.setItem(row, 3, kind_item)
                self.details_table.setItem(row, 4, fave_item)
                self.details_table.setItem(row, 5, tags_item)

            self.details_table.setSortingEnabled(True)
            self.details_table.resizeColumnsToContents()

    ########################################################################
    #                       Selection Handling                              #
    ########################################################################
    #                       Selection Handling                              #
    ########################################################################
    def icon_selection_changed(self):
        selected_items = self.icon_list.selectedItems()
        if not selected_items:
            self.disable_preview_buttons()
            return
        item = selected_items[0]
        self.enable_preview_buttons(item.text())
        self.update_favorite_buttons(item.data(Qt.UserRole))

    def list_selection_changed(self):
        selected_items = self.list_list.selectedItems()
        if not selected_items:
            self.disable_preview_buttons()
            return
        item = selected_items[0]
        self.enable_preview_buttons(item.text())
        self.update_favorite_buttons(item.data(Qt.UserRole))

    def details_selection_changed(self):
        selected_ranges = self.details_table.selectedRanges()
        if not selected_ranges:
            self.disable_preview_buttons()
            return
        row = selected_ranges[0].topRow()
        name_item = self.details_table.item(row, 0)
        if not name_item:
            self.disable_preview_buttons()
            return
        self.enable_preview_buttons(name_item.text())
        self.update_favorite_buttons(name_item.data(Qt.UserRole))
    def disable_preview_buttons(self):
        self.add_favorite_btn.setEnabled(False)
        self.remove_favorite_btn.setEnabled(False)
        self.large_preview_label.clear()
        self.tag_input.setEnabled(False)
    def enable_preview_buttons(self, filename):
        self.add_favorite_btn.setEnabled(True)
        self.remove_favorite_btn.setEnabled(True)
        self.tag_input.setEnabled(True)
        path = self.find_path_by_name(filename)
        self.update_preview(path)

    def update_favorite_buttons(self, path):
        is_fave = self.image_manager.is_favorite(path)
        self.add_favorite_btn.setEnabled(not is_fave)
        self.remove_favorite_btn.setEnabled(is_fave)

    def find_path_by_name(self, name):
        for w in self.wallpapers_data:
            if w["name"] == name:
                return w["path"]
        return None

    def update_preview(self, path):
        if not path:
            self.large_preview_label.clear()
            return
        try:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                self.large_preview_label.clear()
                self.show_error(f"Failed to load image: {path}")
                return
            scaled = pixmap.scaled(
                self.large_preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.large_preview_label.setPixmap(scaled)
        except Exception as e:
            self.large_preview_label.clear()
            self.show_error(f"Error loading preview image: {e}")
    ########################################################################
    #                       Favorites Management                            #
    ########################################################################
    #                       Favorites Management                            #
    ########################################################################
    def add_favorite(self):
        path = self.get_current_selected_path()
        if not path:
            self.show_error("No wallpaper selected.")
            return
        self.image_manager.add_favorite(path)
        self.update_favorite_buttons(path)
        self.refresh_wallpaper_data()
        self.log_message(f"Added to favorites: {os.path.basename(path)}")

    def remove_favorite(self):
        path = self.get_current_selected_path()
        if not path:
            self.show_error("No wallpaper selected.")
            return
        self.image_manager.remove_favorite(path)
        self.update_favorite_buttons(path)
        self.refresh_wallpaper_data()
        self.log_message(f"Removed from favorites: {os.path.basename(path)}")
        if self.current_view == "all":
            # Switch to favorites view.
            self.current_view = "favorites"
            filtered_data = [item for item in self.wallpapers_data if item["favorite"]]
            self.populate_views_with_data(filtered_data)
            self.show_favorites_btn.setText("Show All")
        else:
            # Switch back to all wallpapers.
            self.current_view = "all"
            self.populate_views_with_data(self.wallpapers_data)
            self.show_favorites_btn.setText("Show Favorites")

    def toggle_favorites_view(self):
        """Toggle between showing all wallpapers and showing only favorites."""
        if self.current_view == "all":
            # Switch to favorites view
            self.current_view = "favorites"
            # Filter for favorites only
            filtered_data = [item for item in self.wallpapers_data if item["favorite"]]
            self.populate_views_with_data(filtered_data)
            self.show_favorites_btn.setText("Show All")
        else:
            # Switch back to all wallpapers
            self.current_view = "all"
            # Show all images
            self.populate_views_with_data(self.wallpapers_data)
            self.show_favorites_btn.setText("Show Favorites")
    ########################################################################
    #                           Tags Management                             #
    ########################################################################
    def add_tags_to_current(self):
        path = self.get_current_selected_path()
        if not path:
            self.show_error("No wallpaper selected.")
            return
        tags = self.tag_input.text().strip()
        if tags:
            self.image_manager.add_tags(path, tags)
            self.tag_input.clear()
            self.refresh_wallpaper_data()
            self.log_message(f"Added tags to: {os.path.basename(path)}")

    def get_current_selected_path(self):
        # Check if preview_view_selector exists first
        if hasattr(self, 'preview_view_selector'):
            mode_index = self.preview_view_selector.currentIndex()
            
            # Check for icon view
            if mode_index == 0 and hasattr(self, 'icon_list'):
                items = self.icon_list.selectedItems()
                if items:
                    return items[0].data(Qt.UserRole)
            
            # Check for list view
            elif mode_index == 1 and hasattr(self, 'list_list'):
                items = self.list_list.selectedItems()
                if items:
                    return items[0].data(Qt.UserRole)
            
            # Check for details view
            elif hasattr(self, 'details_table'):
                sel = self.details_table.selectedRanges()
                if sel:
                    row = sel[0].topRow()
                    name_item = self.details_table.item(row, 0)
                    if name_item:
                        return name_item.data(Qt.UserRole)
        else:
            # If preview_view_selector doesn't exist, check each view component directly
            # Check icon view
            if hasattr(self, 'icon_list'):
                items = self.icon_list.selectedItems()
                if items:
                    return items[0].data(Qt.UserRole)
            
            # Check list view
            if hasattr(self, 'list_list'):
                items = self.list_list.selectedItems()
                if items:
                    return items[0].data(Qt.UserRole)
            
            # Check details view
            if hasattr(self, 'details_table'):
                sel = self.details_table.selectedRanges()
                if sel:
                    row = sel[0].topRow()
                    name_item = self.details_table.item(row, 0)
                    if name_item:
                        return name_item.data(Qt.UserRole)
                        
        # If no selection found or views don't exist
        return None

    ########################################################################
    #                        Layout Toggle Methods                          #
    ########################################################################
    def set_horizontal_layout(self):
        self.file_and_preview_splitter.setOrientation(Qt.Horizontal)
        self.file_and_preview_splitter.setSizes([self.width() // 2, self.width() // 2])

    def set_vertical_layout(self):
        self.file_and_preview_splitter.setOrientation(Qt.Vertical)
        self.file_and_preview_splitter.setSizes([self.height() // 2, self.height() // 2])

    ########################################################################
    ########################################################################
    #                       Wallpaper Worker Controls                       #
    ########################################################################
    def start_wallpaper_changer(self):
        """
        Start the wallpaper changing service if it's not already running.
        
        This method:
        1. Checks if the service is already running
        2. Updates UI button states
        3. Creates and configures the worker thread
        4. Logs the action
        """
        if self.wallpaper_worker and self.wallpaper_worker.isRunning():
            self.log_message("Wallpaper changer is already running.")
            return
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

        self.wallpaper_worker = WallpaperWorker(self.config, self.image_manager)
        self.wallpaper_worker.wallpaperChanged.connect(
            lambda path: self.config.update({'last_change': path})
        )
        self.wallpaper_worker.wallpaperChanged.connect(
            lambda path: self.update_statistics()
        )
        self.wallpaper_worker.statusUpdated.connect(self.log_message)
        self.wallpaper_worker.error.connect(self.show_error)
        # Connect the finished signal to clean up resources
        self.wallpaper_worker.finished.connect(self.cleanup_wallpaper_worker)

        self.wallpaper_worker.start()
        # Only change wallpaper immediately if change_on_start is True
        if self.config.get('change_on_start', True):
            self.wallpaper_worker.change_wallpaper()
        self.log_message("Wallpaper changer started.")
    def stop_wallpaper_changer(self):
        if self.wallpaper_worker:
            self.wallpaper_worker.stop()
            if not self.wallpaper_worker.wait(5000):  # 5 second timeout
                self.wallpaper_worker.terminate()
                self.log_message("Force-terminated wallpaper changer.")
            else:
                self.log_message("Wallpaper changer stopped.")
            
            # Clean up to prevent memory leaks
            self.wallpaper_worker.deleteLater()
            self.wallpaper_worker = None
            
            # Process events to ensure proper cleanup
            QApplication.processEvents()
        
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

    def next_wallpaper(self):
        if self.wallpaper_worker:
            self.wallpaper_worker.change_wallpaper()
            
    def cleanup_wallpaper_worker(self):
        """
        Clean up resources when the wallpaper worker thread has finished.
        This method is called when the finished signal is emitted.
        """
        if self.wallpaper_worker:
            # Clean up to prevent memory leaks
            try:
                self.wallpaper_worker.deleteLater()
                self.wallpaper_worker = None
                
                # Update UI buttons to reflect the worker has stopped
                self.stop_btn.setEnabled(False)
                self.start_btn.setEnabled(True)
                
                self.log_message("Wallpaper changer finished.")
            except Exception as e:
                self.show_error(f"Error during wallpaper worker cleanup: {e}")

    ########################################################################
    
    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Choose Wallpaper Directory",
            self.dir_label.text()
        )
        if directory:
            self.dir_label.setText(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Update configuration with the new directory
            self.config['wallpaper_directory'] = directory
            self.save_config()
            # Reinitialize the image manager with the new directory
            self.image_manager = ImageManager(directory)
            self.log_message(f"Changed wallpaper directory to: {directory}")
            self.refresh_wallpaper_data()
        return directory
    def cleanup_directory(self):
        reply = QMessageBox.question(
            self,
            "Confirm Cleanup",
            "Remove duplicates and low-quality images; limit to Max Local Wallpapers if exceeded?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            success = self.image_manager.cleanup_directory(
                max_files=self.max_wallpapers_spin.value()
            )
            if success:
                self.log_message("Cleanup completed successfully.")
            else:
                self.show_error("Error during cleanup.")
            self.update_statistics()
            self.refresh_wallpaper_data()

    def update_statistics(self):
        self.stats_list.clear()
        directory = Path(self.config['wallpaper_directory'])
        total_files = len(list(directory.glob('*')))
        total_size = sum(f.stat().st_size for f in directory.glob('*'))
        shown_today = 0
        if self.wallpaper_worker:
            shown_today = len(self.wallpaper_worker.used_images)

        stats = [
            f"Total wallpapers: {total_files}",
            f"Total size: {total_size / (1024*1024):.2f} MB",
            f"Wallpapers shown today: {shown_today}",
            f"Last change: {self.config.get('last_change', 'Never')}"
        ]
        self.stats_list.addItems(stats)

    ########################################################################
    #                           Scraping Logic                              #
    ########################################################################
    def start_scraping(self):
        if self.scraper_worker and self.scraper_worker.isRunning():
            self.log_message("Scraper is already running.")
            return
        self.save_config()
        
        # Step 1: Start a preview-only scraping worker
        self.log_scraper_message("Fetching image previews...")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
        
        # Create worker in preview-only mode
        self.scraper_worker = MultiSourceScraperWorker(self.config, self.image_manager, preview_only=True)
        self.scraper_worker.statusUpdated.connect(self.log_scraper_message)
        self.scraper_worker.error.connect(self.show_error)
        self.scraper_worker.progress.connect(self.update_scraper_progress)
        
        # Connect to the preview ready signal
        self.scraper_worker.previewReady.connect(self.show_preview_dialog)
        self.scraper_worker.finished.connect(self.preview_fetching_finished)
        
        # Start the worker
        self.scraper_worker.start()
        self.log_scraper_message("Preview fetching started.")
    
    def preview_fetching_finished(self):
        """Called when preview fetching is complete but before download starts"""
        self.log_scraper_message("Preview fetching completed.")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)
    
    def show_preview_dialog(self, preview_data):
        """Shows the preview dialog with the fetched image data"""
        # Log the number of preview items
        logger.debug("Number of preview items: %d", len(preview_data))
        # Create and show the preview dialog
        preview_dialog = ImagePreviewDialog(preview_data, self)
        result = preview_dialog.exec_()
        
        if result == QDialog.Accepted:
            # User clicked OK, get the selected images
            selected_metadata = preview_dialog.get_selected_metadata()
            
            if not selected_metadata:
                self.log_scraper_message("No images selected for download.")
                return
            
            self.log_scraper_message(f"Downloading {len(selected_metadata)} selected images...")
            
            # Step 2: Start downloading the selected images
            download_dir = self.config['wallpaper_directory']
            
            # Reset progress bar before starting download
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(0)
            total_count = len(selected_metadata)
            downloaded_count = 0
            
            for idx, metadata in enumerate(selected_metadata):
                try:
                    # Update progress
                    progress = min(100, int(100 * idx / total_count))
                    if hasattr(self, 'progress_bar'):
                        self.progress_bar.setValue(progress)
                    
                    # Get image URL and filename
                    image_url = metadata.get('url')
                    filename = metadata.get('filename')
                    
                    if not image_url or not filename:
                        continue
                    
                    # Use Path for consistent cross-platform file handling
                    file_path = Path(download_dir) / filename
                    
                    # Skip if file already exists
                    if file_path.exists():
                        self.log_scraper_message(f"File already exists: {filename}")
                        continue
                    
                    # Download the image
                    self.log_scraper_message(f"Downloading: {filename}")
                    image_response = requests.get(image_url, stream=True)
                    if image_response.status_code == 200:
                        with self._thread_safe():
                            if not file_path.exists():
                                with open(str(file_path), 'wb') as f:
                                    for chunk in image_response.iter_content(1024):
                                        f.write(chunk)
                        
                        # file_path is already a Path object, no need to convert again
                        self.image_manager.add_image_metadata(str(file_path), {
                            'source': metadata.get('source', 'unknown'),
                            'tags': metadata.get('tags', []),
                            'date_added': datetime.now().isoformat()
                        })
                        
                        downloaded_count += 1
                    else:
                        self.log_scraper_message(f"Failed to download: {filename} (Status: {image_response.status_code})")
                
                except Exception as e:
                    self.show_error(f"Error downloading image: {str(e)}")
            
            # Update final progress and refresh the display
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(100)
            self.log_scraper_message(f"Downloaded {downloaded_count} of {total_count} selected images.")
            self.refresh_wallpaper_data()
            
        else:
            # User cancelled
            self.log_scraper_message("Download cancelled by user.")
    def scraper_finished(self):
        self.log_scraper_message("Scraper finished.")
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(0)

    def log_scraper_message(self, text):
        logger.info(text)
        self.scraper_log.addItem(text)
        self.scraper_log.scrollToBottom()

    def update_scraper_progress(self, value):
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)

    ########################################################################
    #                             Logging & Errors                          #
    ########################################################################
    def log_message(self, text):
        logger.info(text)
        self.log_list.addItem(text)
        self.log_list.scrollToBottom()

    def show_error(self, text):
        logger.error(text)
        QMessageBox.critical(self, "Error", text)

    ########################################################################
    #                           System Tray Setup                           #
    ########################################################################
    def setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        icon_path = self.config.get('tray_icon_path', os.path.join(os.path.dirname(__file__), "wallpaper-menu-bar-icon.png"))
        if Path(icon_path).exists():
            self.tray_icon.setIcon(QIcon(icon_path))
        else:
            self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show GUI")
        show_action.triggered.connect(self.show)

        next_action = tray_menu.addAction("Next Wallpaper")
        next_action.triggered.connect(self.next_wallpaper)

        open_action = tray_menu.addAction("Open Wallpaper Folder")
        open_action.triggered.connect(self.open_wallpaper_folder)

        tray_menu.addSeparator()

        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    ########################################################################
    #                           Close Event                                 #
    ########################################################################
    def open_wallpaper_folder(self):
        """Open the wallpaper directory using platform-specific commands."""
        try:
            directory = self.config['wallpaper_directory']
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', directory])
            elif platform.system() == 'Windows':  # Windows
                subprocess.run(['explorer', directory])
            elif platform.system() == 'Linux':  # Linux
                subprocess.run(['xdg-open', directory])
            else:
                self.show_error(f"Unsupported platform: {platform.system()}")
        except Exception as e:
            self.show_error(f"Error opening folder: {e}")

    def closeEvent(self, event):
        if self.tray_icon.isVisible():
            self.hide()
            if self.config.get('show_notifications', True):
                self.tray_icon.showMessage(
                    "Wallpaper Manager",
                    "Application minimized to menu bar",
                    QSystemTrayIcon.Information,
                    self.config.get('notification_duration', 3) * 1000
                )
            event.ignore()
        else:
            # Clean up resources before closing
            # Stop the refresh timer
            if hasattr(self, 'refresh_timer') and self.refresh_timer.isActive():
                self.refresh_timer.stop()

            # Clean up worker threads
            if hasattr(self, 'wallpaper_worker') and self.wallpaper_worker:
                self.wallpaper_worker.stop()
                if not self.wallpaper_worker.wait(3000):
                    self.wallpaper_worker.terminate()
                self.wallpaper_worker.deleteLater()
                self.wallpaper_worker = None
                
            if hasattr(self, 'scraper_worker') and self.scraper_worker:
                self.scraper_worker.stop()
                if not self.scraper_worker.wait(3000):
                    self.scraper_worker.terminate()
                self.scraper_worker.deleteLater()
                self.scraper_worker = None
                
            if hasattr(self, 'scan_worker') and self.scan_worker:
                if self.scan_worker.isRunning():
                    self.scan_worker.terminate()
                    self.scan_worker.wait()
                self.scan_worker.deleteLater()
                self.scan_worker = None
                    
            # Clean up image cache
            if hasattr(self, 'image_manager'):
                self.image_manager.hash_cache.clear()
                
            # Release mutex if locked
            if hasattr(self, 'mutex') and self.mutex.tryLock():
                self.mutex.unlock()
                
            # Save any pending configuration changes
            self.save_config()
                
            # Ensure all operations are finished
            QApplication.processEvents()
                
            event.accept()


#                                Main Entry                                   #
###############################################################################
if __name__ == '__main__':
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    # Modern look
    app.setStyleSheet("""
        QMainWindow {
            background-color: #353535;
        }
        QWidget {
            font-size: 10pt;
        }
        QPushButton {
            background-color: #2a82da;
            border: none;
            color: white;
            padding: 5px 15px;
            border-radius: 3px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #3292ea;
        }
        QPushButton:pressed {
            background-color: #1a72ca;
        }
        QPushButton:disabled {
            background-color: #555555;
        }
        QGroupBox {
            border: 1px solid #555555;
            border-radius: 5px;
            margin-top: 1em;
            padding-top: 10px;
        }
        QGroupBox::title {
            color: white;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QListWidget, QTableWidget {
            background-color: #252525;
            border: 1px solid #555555;
        }
        QLineEdit {
            background-color: #252525;
            border: 1px solid #555555;
            padding: 5px;
            border-radius: 3px;
            color: white;
        }
        QLineEdit:focus {
            border: 1px solid #2a82da;
        }
        QSpinBox, QTimeEdit {
            background-color: #252525;
            border: 1px solid #555555;
            padding: 3px;
            color: white;
        }
        QLabel {
            color: white;
        }
        QCheckBox {
            color: white;
        }
        QMessageBox {
            background-color: #353535;
        }
        QMessageBox QLabel {
            color: white;
        }
        QMessageBox QPushButton {
            width: 60px;
            padding: 5px 15px;
        }
        QMenu {
            background-color: #353535;
            color: white;
            border: 1px solid #555555;
        }
        QMenu::item {
            padding: 5px 20px;
        }
        QMenu::item:selected {
            background-color: #2a82da;
        }
        QMenu::separator {
            height: 1px;
            background-color: #555555;
            margin: 5px 0px;
        }
        QTabWidget::pane {
            border: 1px solid #555555;
        }
        QTabBar::tab {
            background-color: #353535;
            color: white;
            border: 1px solid #555555;
            padding: 5px 10px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #2a82da;
        }
        QTabBar::tab:hover {
            background-color: #3292ea;
        }
        QComboBox {
            background-color: #252525;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 5px;
            color: white;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            image: none;
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #252525;
            border: 1px solid #555555;
            color: white;
            selection-background-color: #2a82da;
        }
        QTableWidget {
            background-color: #252525;
            border: 1px solid #555555;
            gridline-color: #555555;
        }
        QHeaderView::section {
            background-color: #353535;
            color: white;
            border: none;
            padding: 4px;
        }
        QScrollBar:vertical {
            border: none;
            background-color: #353535;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background-color: #555555;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #666666;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            border: none;
            background-color: #353535;
            height: 10px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background-color: #555555;
            min-width: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #666666;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        QToolButton {
            background-color: #2a82da;
            border: none;
            color: white;
            padding: 5px;
            border-radius: 3px;
        }
        QToolButton:hover {
            background-color: #3292ea;
        }
        QToolButton:pressed {
            background-color: #1a72ca;
        }
    """)

    window = MainWindow()
    if not window.config.get('minimize_to_tray_on_start', True):
        window.show()

    sys.exit(app.exec_())
