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
    path('propose', views.propose, name='propose'),
    path('validated', views.validated, name='validated'),
    path('to_validate', views.to_validate, name='to_validate'),
    path('users', views.users, name='users'),
    path('chatbot/<str:chat_id>/edit_name/', views.edit_chat_name, name='edit_chat_name'),
    
    ]
