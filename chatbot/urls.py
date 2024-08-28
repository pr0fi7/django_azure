from django.urls import path, re_path
from . import views

urlpatterns = [
    path('chatbot/<str:chat_id>/', views.chatbot, name='chatbot'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name='logout'),
    path('chats', views.chat_list_view, name='chats'),
    path('start-chat', views.start_chat, name='start_chat'),
    path('chatbot/<str:chat_id>/clear_chat/', views.clear_chat, name='clear_chat'),
    path('chatbot/<str:chat_id>/clear_all/', views.clear_all, name='clear_all'),

    ]
