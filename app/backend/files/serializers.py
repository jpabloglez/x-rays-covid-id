from rest_framework import serializers

class ImageSerializer(serializers.Serializer):
    image = serializers.ImageField()
    # url = serializers.CharField(max_length=255, required=False)

