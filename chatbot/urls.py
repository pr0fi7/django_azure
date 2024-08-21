from django.urls import path, re_path
from . import views

urlpatterns = [
    path('chatbot/<str:chat_id>/', views.chatbot, name='chatbot'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('<str:chat_id>/upload/', views.upload, name='upload'),
    path('chats', views.chat_list_view, name='chats'),
    path('start-chat', views.start_chat, name='start_chat'),
    path('chats/create/', views.create_chat_view, name='create_chat'),
]
