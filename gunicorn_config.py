import multiprocessing
import os

port = os.environ.get("PORT", 8000)
bind = f"0.0.0.0:{port}"
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
timeout = 120
