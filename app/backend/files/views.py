from django.shortcuts import render

# Create your views here.
import os
import os.path as op
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser

from .serializers import ImageSerializer
from pathlib import Path
from django.conf import settings


class ImageUploadView(APIView):
    parser_classes = [MultiPartParser]
    def post(self, request, format=None):
        serializer = ImageSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            # image_path = op.join(settings.MEDIA_ROOT, image.name)
            image_path = op.join('/var/www/static', image.name)
            image_path = str(image_path)
            print("image", image)
            print("image_path", image_path)
            serializer.validated_data['image'] = image_path
            print("serializer.data", serializer.data)
            print("serializer.validated_data", serializer.validated_data)
            print('imageUrl', serializer.data['image'])
            with open(image_path, 'wb') as file:
                file.write(image.read())
            # Update image path to use the static url
            image_path = op.join('static/', image.name)

            return Response(
                {
                    'detail': 'Image uploaded successfully',
                    'imageUrl': image_path,
                    #'data': serializer.data
                 }, 
                status=200
            )
        return Response(
            {
                'detail': 'Image upload failed',
                'imageUrl': '',
                #"data": serializer.errors, 
            },
            status=400
        )
