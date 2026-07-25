"""backend URL configuration.

https://docs.djangoproject.com/en/4.2/topics/http/urls/
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from files.views import ImageUploadView

# The users app keeps its models, because AUTH_USER_MODEL points at them, but
# its views are gone: they exposed unauthenticated list/update/delete over every
# account. Phase B rebuilds that surface on FastAPI with real authentication.
urlpatterns = [
    path("admin/", admin.site.urls),
    path("files/", ImageUploadView.as_view()),
]

if settings.DEBUG:
    # Development only. Behind a real deployment the media directory is served
    # by the web server, not by Django.
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
