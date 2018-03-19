"""
WSGI config for BotHelperOffline project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not PROJECT_DIR in sys.path:
    sys.path.insert(0, PROJECT_DIR)

sys.path.append('/newdisk1/csbot/BotHelper/BotHelperOffline/BotHelperOffline')
sys.path.append('/newdisk1/csbot/BotHelper/BotHelperOffline')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BotHelperOffline.settings")

application = get_wsgi_application()
