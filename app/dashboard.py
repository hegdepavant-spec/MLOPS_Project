"""
Compatibility wrapper for dashboard/app.py
"""
import os
import sys

# Ensure dashboard path is available
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dashboard_file = os.path.join(BASE_DIR, 'dashboard', 'app.py')

with open(dashboard_file, 'r', encoding='utf-8') as f:
    code = f.read()

exec(code)
