"""Settings used by the test suite.

pytest-django reads DATABASES before any conftest runs, so the environment the
real settings module requires has to be in place at import time — hence a
module rather than a fixture.
"""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-key-never-used-outside-pytest")
os.environ.setdefault("DJANGO_DEBUG", "false")

from backend.settings import *

# Never touch the developer's SQLite file from a test run.
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}

# DEBUG is off above so the suite exercises the production configuration, but
# the test client speaks plain HTTP: leave the redirect on and every request
# answers 301 before reaching a view.
SECURE_SSL_REDIRECT = False
