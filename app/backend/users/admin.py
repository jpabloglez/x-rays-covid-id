from django.contrib import admin

from users.models import Organization, User, UserProfile

admin.site.register(User)
admin.site.register(UserProfile)
admin.site.register(Organization)
