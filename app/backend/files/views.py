from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ImageSerializer
from pathlib import Path
from django.conf import settings


class ImageUploadView(APIView):
    def post(self, request, format=None):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            image_path = Path.joinpath(settings.MEDIA_ROOT, image.name)
            print("image", image)
            print("image_path", image_path)
            
            with open(image_path, 'wb') as file:
                file.write(image.read())

            return Response(
                {
                    'message': 'Image uploaded successfully',
                    'imageUrl': serializer.data['image']
                 }, 
                status=200)
        return Response(serializer.errors, status=400)
