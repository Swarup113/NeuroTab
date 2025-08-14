# gunicorn.conf.py
bind = "0.0.0.0:8080"
workers = 1  # Free tier limit
timeout = 180  # Increased to 180 seconds for slow computations