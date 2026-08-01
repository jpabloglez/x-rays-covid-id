"""Django URL configuration -- serving nothing.

The HTTP surface moved to FastAPI in XRAYS-09: `api.main` serves uploads and
inference, and compose runs uvicorn rather than `manage.py runserver`. What is
left here is the `users` app, which exists only so its models and migrations
survive as the reference for the port in XRAYS-10, and `AUTH_USER_MODEL` still
has to resolve for `manage.py makemigrations` to work at all.

Nothing routes to a view. When the users port lands, this project goes with it.
"""

urlpatterns = []
