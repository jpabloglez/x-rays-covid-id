from django.apps import AppConfig
from django.contrib import admin

class UsersProfilesConfig(AppConfig):
    name = 'usersprofiles'

    def ready(self):
        # Import the signals module here so that it is registered
        # with Django when the app is ready.
        import users.signals

# Register your models here.
from users.models import User, UserProfile, Organization

admin.site.register(User)
admin.site.register(UserProfile)
admin.site.register(Organization)


