from rest_framework import serializers
from users.models import (
    User,
    UserProfile
)
from datetime import datetime
from django.utils import timezone
from django.utils.timesince import timesince

LANGUAGE_CHOICES = (
    ('en', 'English'),
    ('es', 'Spanish'),
)


class UserSerializer(serializers.ModelSerializer):

    #time_since_joined = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        #fields = ('id', 'email')
        #fields = '__all__'
        exclude = ('id', 'password')

    # def get_time_since_joined(self, object):
    #     join_data = object.created_at
    #     return timesince(join_data, datetime.now())

# class UserManagerSerializer(serializers.ModelSerializer):

#     days_active = serializers.SerializerMethodField()

#     class Meta:
#         model = UserManager
#         fields = '__all__'

#     def get_days_active(self, object):
#         join_data = object.created_at
#         return timesince(join_data, datetime.now())

class UserProfileSerializer(serializers.ModelSerializer):

    user = serializers.StringRelatedField(read_only=True)
    email = serializers.StringRelatedField(read_only=True, source='user.email')
    # role = serializers.StringRelatedField(read_only=True, source='user.role')
    time_since_joined = serializers.SerializerMethodField()
        
    class Meta:
        model = UserProfile
        #fields = ('id', 'name', 'email', 'phone', 'address', 
        # 'city', 'state', 'country', 'zip', 'image', 'created_at', 'updated_at', 'deleted_at')
        exclude = ('id',)
        # fields = '__all__'

    def get_time_since_joined(self, object):
        join_data = object.created_at
        return timesince(join_data, timezone.now())

    # Add validation to the serializer
    def validate(self, data):
        """ Check the provided language is valid """
        if data['language'] not in dict(LANGUAGE_CHOICES).keys():
            raise serializers.ValidationError("The provided language is not valid.")
        return data