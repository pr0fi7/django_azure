import os
from django.core.wsgi import get_wsgi_application

settings_module = 'django_chatbot.deployment' if 'WEBSITE_HOSTNAME' in os.environ else 'django_chatbot.settings'

os.environ.setdefault('DJANGO_SETTINGS_MODULE', settings_module)

application = get_wsgi_application()