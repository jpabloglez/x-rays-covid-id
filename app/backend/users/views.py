from django.shortcuts import render
from django.utils import timezone

# Create your views here.

from users.models import (
    User,
    UserProfile
)
from users.serializers import (
    UserSerializer,
    #UserManagerSerializer,
    UserProfileSerializer
)

from django.contrib.auth.models import User

from rest_framework import status
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from users.permissions import IsAdminUserOrReadOnly

#class UserViewSet(viewsets.ModelViewSet):

   
class UserListView(APIView):
    __doc__ = """ User List View """

    def get(self, request, format=None):
        users = User.objects.filter(is_active=True)
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class UserDetailView(APIView):
    __doc__ = """ User Detail View """

    def get(self, request, pk, format=None):
        user = User.objects.get(pk=pk)
        serializer = UserSerializer(user)
        return Response(serializer.data, status=status.HTTP_200_OK)


    def put(self, request, pk, format=None):
        user = User.objects.get(pk=pk)
        serializer = UserSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        user = User.objects.get(pk=pk)
        # user.delete()
        user.deleted_at = timezone.now()
        user.is_active = False
        return Response(status=status.HTTP_204_NO_CONTENT)

class UserProfileDetail(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, format=None):
        user = request.user
        serializer = UserProfileSerializer(user.userprofile)
        return Response(serializer.data, status.HTTP_200_OK)

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserProfile.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class UserProfileListView(APIView):
    __doc__ = """ User Profile List View """
    permission_classes = [IsAuthenticated]

    def get(self, request, format=None):
        #user = request.user
        #serializer = UserProfileSerializer(user.userprofile)
        users = UserProfile.objects.all()
        serializer = UserProfileSerializer(users, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request, format=None):
        serializer = UserProfileSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserProfileDetailView(APIView):
    __doc__ = """ User Profile Detail View """

    permission_classes = [IsAuthenticated]

    def get_object(self, pk):
        # try:
        #     return UserProfile.objects.get(pk=pk)
        # except UserProfile.DoesNotExist:
        #     raise(status.HTTP_404_NOT_FOUND)
        return UserProfile.objects.get(pk=pk)


    def get(self, request, pk, format=None):
        user = self.get_object(pk)
        serializer = UserProfileSerializer(user)
        return Response(serializer.data, status=status.HTTP_200_OK)

        #user = UserProfile.objects.get(pk=pk)
        #serializer = UserProfileSerializer(user)
        #return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, format=None):
        user = self.get_object(pk)
        serializer = UserProfileSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        """ Soft deletion of the user profile """
        user = self.get_object(pk)
        user.deleted_at = timezone.now()
        user.save()
        return Response(status=status.HTTP_204_NO_CONTENT)
        # Complete deletion of the user profile
        #user = self.get_object(pk)
        #user.delete()
        #return Response(status=status.HTTP_204_NO_CONTENT)