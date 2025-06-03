import json
from http.server import BaseHTTPRequestHandler
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.dashboard import app

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Get the Dash app's HTML
        html = app.index()
        
        # Set response headers
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Send the HTML response
        self.wfile.write(html.encode())
        return 