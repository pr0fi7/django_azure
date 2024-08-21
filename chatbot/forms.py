from django import forms
from .models import Chat

class DocumentUploadForm(forms.Form):
    document = forms.FileField()

class ChatForm(forms.ModelForm):
    class Meta:
        model = Chat
        fields = '__all__'

