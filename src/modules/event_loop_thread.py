
import asyncio
import threading


class EventLoopThread:
    """Manages a persistent event loop in a background thread for MCP connections."""
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self.ctx = None
        
    def start(self):
        """Start the event loop in a background thread."""
        if self.thread is not None and self.thread.is_alive():
            return
        
        # Capture the current Streamlit script context
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
            self.ctx = get_script_run_ctx()
        except ImportError:
            self.ctx = None
            
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        
        # Add script context to the thread if available
        if self.ctx is not None:
            try:
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                add_script_run_ctx(self.thread, self.ctx)
            except Exception:
                pass
                
        self.thread.start()
        
    def _run_loop(self):
        """Run the event loop in the background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        
    def run_coroutine(self, coro):
        """Run a coroutine in the background event loop and return result."""
        if self.loop is None:
            self.start()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
        
    def stop(self):
        """Stop the event loop and thread."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)