import json
from http.server import BaseHTTPRequestHandler
import sys
import os
import dash
from dash import html

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.visualization.dashboard import app

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Get the Dash app's HTML
            html = app.index()
            
            # Set response headers
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Send the HTML response
            self.wfile.write(html.encode())
            return
        except Exception as e:
            # Log the error
            print(f"Error in serverless function: {str(e)}")
            
            # Return error response
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            error_html = html.Div([
                html.H1("Error Loading Dashboard"),
                html.P(f"An error occurred: {str(e)}"),
                html.P("Please try refreshing the page or contact support if the issue persists.")
            ])
            self.wfile.write(str(error_html).encode())
            return 