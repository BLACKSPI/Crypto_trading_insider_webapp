import json
from http.server import BaseHTTPRequestHandler
import sys
import os
import dash
from dash import html
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.visualization.dashboard import app
    logger.info("Successfully imported dashboard app")
except Exception as e:
    logger.error(f"Error importing dashboard app: {str(e)}")
    raise

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            logger.info(f"Handling request for path: {self.path}")
            
            # Handle different types of requests
            if self.path.startswith('/_dash-assets/'):
                # Serve static assets
                asset_path = self.path.replace('/_dash-assets/', '')
                try:
                    with open(os.path.join('src/visualization/assets', asset_path), 'rb') as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/javascript')
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception as e:
                    logger.error(f"Error serving asset {asset_path}: {str(e)}")
                    self.send_error(404)
                    return
            
            # Get the Dash app's HTML
            try:
                html = app.index()
                logger.info("Successfully generated dashboard HTML")
            except Exception as e:
                logger.error(f"Error generating dashboard HTML: {str(e)}")
                raise
            
            # Set response headers
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            # Send the HTML response
            self.wfile.write(html.encode())
            logger.info("Successfully sent HTML response")
            return
            
        except Exception as e:
            logger.error(f"Error in serverless function: {str(e)}")
            
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

    def log_message(self, format, *args):
        """Override to use our logger instead of print"""
        logger.info("%s - %s", self.address_string(), format % args) 