# Generated by Django 5.1 on 2024-08-19 10:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0002_chat_chat_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='chat',
            name='chat_id',
            field=models.TextField(default='default'),
        ),
    ]
