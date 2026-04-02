"""
UI Tools Config Controller
Manages UIToolsConfig CRD reconciliation and watchers.
"""

import logging
import threading
import time
from typing import Optional
from kubernetes import watch

from ..services.ui_tools.loader import (
    _init_k8s_client,
    reload_ui_tools_config,
    clear_ui_tools_config,
    NAMESPACE,
    GROUP,
    VERSION,
    PLURAL,
)


class UIToolsWatcher:
    """Manager for watching UIToolsConfig CRD changes and syncing the registry."""
    
    def __init__(self):
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_running = False
    
    def _watch_ui_tools_config(self):
        """
        Watch for changes to UIToolsConfig resources and update the registry.
        
        Runs in a background thread and monitors all UIToolsConfig resources in the namespace.
        When changes are detected, reloads the affected config into the registry.
        """
        retry_delay = 5  # Start with 5 second delay
        max_retry_delay = 60  # Max 60 seconds
        
        while self._watcher_running:
            try:
                api = _init_k8s_client()
                watcher_client = watch.Watch()
                
                logging.info("Starting UIToolsConfig watcher")
                
                # Watch for changes to UIToolsConfig resources
                for event in watcher_client.stream(
                    api.list_namespaced_custom_object,
                    group=GROUP,
                    version=VERSION,
                    namespace=NAMESPACE,
                    plural=PLURAL,
                ):
                    if not self._watcher_running:
                        break
                    
                    event_type = event['type']  # ADDED, MODIFIED, DELETED
                    resource = event['object']
                    resource_name = resource.get('metadata', {}).get('name', 'unknown')
                    
                    try:
                        if event_type in ['ADDED', 'MODIFIED']:
                            logging.debug(f"UIToolsConfig '{resource_name}' {event_type.lower()}, reloading...")
                            reload_ui_tools_config(resource)
                        
                        elif event_type == 'DELETED':
                            logging.debug(f"UIToolsConfig '{resource_name}' deleted, clearing registry...")
                            clear_ui_tools_config(resource_name)
                    
                    except Exception as e:
                        logging.error(f"Error processing UIToolsConfig event for '{resource_name}': {e}")
                
                # If we exit the stream, reset retry delay for next attempt
                retry_delay = 5
                
            except Exception as e:
                logging.error(f"Error in UIToolsConfig watcher: {e}")
                if self._watcher_running:
                    logging.info(f"Watcher will retry in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)
    
    def start(self):
        """Start the UIToolsConfig watcher in a background thread."""
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            logging.debug("UIToolsConfig watcher already running")
            return
        
        self._watcher_running = True
        self._watcher_thread = threading.Thread(target=self._watch_ui_tools_config, daemon=True)
        self._watcher_thread.start()
        logging.info("Started UIToolsConfig watcher thread")
    
    def stop(self):
        """Stop the UIToolsConfig watcher."""
        self._watcher_running = False
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5)
        logging.info("Stopped UIToolsConfig watcher")


def create_ui_tools_watcher() -> UIToolsWatcher:
    """
    Create a UIToolsWatcher instance.
    
    Returns:
        UIToolsWatcher: A new watcher instance ready to be started
    """
    return UIToolsWatcher()
