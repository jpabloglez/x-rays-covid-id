from rest_framework import permissions

class IsAdminUserOrReadOnly(permissions.IsAdminUser):
    """
    The request is authenticated as an admin, or is a read-only request.
    """

    def has_permission(self, request, view):
        is_admin = super().has_permission(request, view)
        return bool(
            request.method in permissions.SAFE_METHODS or
            is_admin
        )

class IsUSerOrReadOnly(permissions.IsAuthenticated):
    """
    The request is authenticated as an admin, or is a read-only request.
    """

    def has_permission(self, request, view):
        if request.user.role == 'user':
            is_user = True
        return bool(
            request.method in permissions.SAFE_METHODS or
            is_user
        )

class IsAdminUserOrReadOnly(permissions.IsAdminUser):
    """
    The request is authenticated as an admin, or is a read-only request.
    """

    def has_permission(self, request, view):
        if request.user.role == 'admin':
            is_admin = True
        return bool(
            request.method in permissions.SAFE_METHODS or
            is_admin
        )

class IsSuperUserOrReadOnly(permissions.IsAdminUser):
    """
    The request is authenticated as an admin, or is a read-only request.
    """

    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.user == request.user

    def has_permission(self, request, view):
        if request.user.role == 'superuser':
            is_superuser = True
        return bool(
            request.method in permissions.SAFE_METHODS or
            is_superuser
        )