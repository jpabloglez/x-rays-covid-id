from django.urls import path
from rest_framework.routers import DefaultRouter
from users.views import (
    UserListView,
    UserDetailView,
    UserProfileViewSet,
    UserProfileListView,
    UserProfileDetailView,
)

# Using router
#router = DefaultRouter()
# router.register(r'users', UserProfileViewSet, basename='users')
#router.register(r'users', UserProfileViewSet, basename='users')
#router.register(r'users/<int:pk>', UserProfileDetailView, basename='users')
#urlpatterns = router.urls

urlpatterns = [
    path(
        '', 
        UserListView.as_view(), 
        name='users-list'),
    path(
        '<int:pk>/',
        UserDetailView.as_view(),
        name='users-detail'),
    path(
        'profile/', 
        UserProfileListView.as_view(), 
        name='users-profile-list'),
    path(
        'profile/<int:pk>/', 
        UserProfileDetailView.as_view(), 
        name='users-profile-detail')
]

