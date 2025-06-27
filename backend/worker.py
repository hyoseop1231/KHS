# backend/worker.py
# This file can be a reference point for Celery but is not strictly necessary
# if the Celery command correctly points to the Celery app instance.

# For example, if you run `celery -A app.tasks:celery_app worker ...`,
# Celery will look for the `celery_app` instance within the `app.tasks` module.

# You can optionally import the app here if this worker.py module itself
# is specified in the -A flag, e.g., `celery -A worker:celery_app worker ...`
# from app.tasks import celery_app

# print("Celery worker module (worker.py) loaded.")
# print("Ensure your Celery command correctly specifies the application instance, e.g., -A app.tasks:celery_app")
