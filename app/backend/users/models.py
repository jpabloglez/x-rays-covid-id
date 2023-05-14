from django.db import models
from django.utils import timezone
from django.contrib.auth.models import (
    AbstractBaseUser, 
    BaseUserManager, 
    PermissionsMixin
)

# Create your models here.
from django.contrib.auth.models import User

ROLES = (
    (1, 'User'),
    (2, 'Guest'),
    (3, 'Admin'),
    (4, 'Manager'),
    (5, 'Superuser')
)


class UserManager(BaseUserManager):

    def _create_user(self, email, password, is_active, is_staff, is_superuser, **extra_fields):
        now = timezone.now()
        user = self.model(
            email=email,
            is_staff=is_staff,
            is_active=is_active,
            is_superuser=is_superuser,
            last_login=now,
            **extra_fields
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        return self._create_user(email, password, True, False, False, **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        user = self._create_user(email, password, True, True, True, **extra_fields)
        return user

    def get_queryset(self):
        return super(UserManager, self).get_queryset()

class User(AbstractBaseUser):
    """ User Model """
    email = models.EmailField(unique=True)
    last_login = models.DateTimeField(default=timezone.now)
    role = models.PositiveBigIntegerField(choices=ROLES, default=1)

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    #REQUIRED_FIELDS = []

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        """ Does the user have a specific permission? """
        return True

    def has_module_perms(self, app_label):
        """ Does the user have permissions to view the app `app_label`? """
        return True

    # @property
    # def is_staff(self):
    #     """ Is the user a member of staff? """
    #     return self.is_staff

    # @property
    # def is_active(self):
    #     """ Is the user active? """
    #     return self.is_active

    # @property
    # def is_superuser(self):
    #     """ Is the user a admin member? """
    #     return self.is_superuser


class Organization(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    centre = models.CharField(max_length=80)
    address = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    billing_address = models.CharField(max_length=100)
    billing_code = models.CharField(max_length=100)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.centre

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, null=True)
    #organization = models.ForeignKey(
    #    Organization,
    #    on_delete=models.CASCADE, 
    #    related_name='userprofile',
    #    null=True,)
    first_name = models.CharField(max_length=80)
    last_name = models.CharField(max_length=80, null=True)
    email = models.EmailField(max_length=80)
    phone = models.CharField(max_length=12, blank=True)
    address = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    zip = models.CharField(max_length=10, blank=True)
    language = models.CharField(max_length=24, default='es')
    image = models.ImageField(upload_to='images/users', blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f'{self.first_name} {self.last_name}'

