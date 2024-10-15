from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Chat(models.Model):
    chat_id = models.CharField(max_length=100)  # Use CharField for text-based IDs
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    source = models.TextField(null=True, blank=True)
    file_name = models.CharField(max_length=255, null=True, blank=True)  
    preprompt = models.TextField(null=True, blank=True)
    additional_preprompt = models.TextField(null=True, blank=True)  
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'
    

class UserSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    chat_id = models.OneToOneField(Chat, on_delete=models.CASCADE, null=True, blank=True)
    role = models.CharField(max_length=100, default='default')
    creativity = models.FloatField(default=0.5)
    length = models.IntegerField(default=0)
    bullet_points = models.IntegerField(default=0)
    chunk_size = models.IntegerField(default=600)
    chunk_overlap = models.IntegerField(default=200)
    num_docs = models.IntegerField(default=5)
    show_source = models.BooleanField(default=True)



    def __str__(self):
        return f'{self.user.username} settings'
    
    