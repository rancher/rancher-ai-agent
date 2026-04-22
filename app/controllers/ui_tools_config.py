"""
UI Tools Config Controller
Manages UIToolsConfig ConfigMaps in Kubernetes, watches for changes, and updates the UI tools registry accordingly.
"""

import logging
import threading
import time
from typing import Optional
from kubernetes import client, config, watch
from ..services.ui_tools.loader import reload_ui_tools_config, clear_ui_tools_config

NAMESPACE = "cattle-ai-agent-system"
UI_TOOLS_LABEL = "app=rancher-ai-ui-tools"

def _init_k8s_client():
    """Initialize Kubernetes client."""
    try:
        # Try in-cluster config first
        config.load_incluster_config()
    except config.ConfigException:
        # Fall back to kubeconfig
        config.load_kube_config()
    
    return client.CoreV1Api()

class UIToolsWatcher:
    """Manager for watching UIToolsConfig ConfigMap changes and syncing the registry."""
    
    def __init__(self):
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_running = False
    
    def _watch_ui_tools_config(self):
        """
        Watch for changes to UI tools ConfigMaps and update the registry.
        
        Runs in a background thread and monitors all ConfigMaps with the rancher-ai-ui-tools label.
        When changes are detected, reloads the affected config into the registry.
        """
        retry_delay = 5  # Start with 5 second delay
        max_retry_delay = 60  # Max 60 seconds
        
        while self._watcher_running:
            try:
                api = _init_k8s_client()
                watcher_client = watch.Watch()
                
                logging.info(f"Starting UI tools ConfigMap watcher (watching for label: {UI_TOOLS_LABEL})")
                
                # Watch for changes to ConfigMaps with the UI tools label
                for event in watcher_client.stream(
                    api.list_namespaced_config_map,
                    namespace=NAMESPACE,
                    label_selector=UI_TOOLS_LABEL,
                ):
                    if not self._watcher_running:
                        break
                    
                    event_type = event['type']  # ADDED, MODIFIED, DELETED
                    resource = event['object']
                    resource_name = resource.metadata.name if resource.metadata else 'unknown'
                    
                    try:
                        if event_type in ['ADDED', 'MODIFIED']:
                            logging.debug(f"UI tools ConfigMap '{resource_name}' {event_type.lower()}, reloading...")
                            
                            resource_dict = {
                                'metadata': {
                                    'name': resource.metadata.name,
                                    'namespace': resource.metadata.namespace,
                                },
                                'data': resource.data or {}
                            }
                            reload_ui_tools_config(resource_dict)
                        
                        elif event_type == 'DELETED':
                            logging.debug(f"UI tools ConfigMap '{resource_name}' deleted, clearing registry...")
                            clear_ui_tools_config(resource_name)
                    
                    except Exception as e:
                        logging.error(f"Error processing UI tools ConfigMap '{resource_name}': {e}")
                
                # If we exit the stream, reset retry delay for next attempt
                retry_delay = 5
                
            except Exception as e:
                logging.error(f"Error in UI tools ConfigMap watcher: {e}")
                if self._watcher_running:
                    logging.info(f"Watcher will retry in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)
    
    def start(self):
        """Start the UI tools ConfigMap watcher in a background thread."""
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            logging.debug("UI tools ConfigMap watcher already running")
            return
        
        self._watcher_running = True
        self._watcher_thread = threading.Thread(target=self._watch_ui_tools_config, daemon=True)
        self._watcher_thread.start()
        logging.info("Started UI tools ConfigMap watcher thread")
    
    def stop(self):
        """Stop the UI tools ConfigMap watcher."""
        self._watcher_running = False
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5)
        logging.info("Stopped UI tools ConfigMap watcher")


def create_ui_tools_watcher() -> UIToolsWatcher:
    """
    Create a UIToolsWatcher instance for monitoring UI tools ConfigMaps.
    
    Returns:
        UIToolsWatcher: A new watcher instance ready to be started
    """
    return UIToolsWatcher()
