from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/new-game/', views.new_game, name='new_game'),
    path('api/move/', views.make_move, name='make_move'),
    path('api/legal-moves/', views.get_legal_moves_view, name='legal_moves'),
]
