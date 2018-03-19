from django.conf.urls import url
import views
urlpatterns = [
    url(r'^init', views.init),
    url(r'^trainAble', views.isable_train),
    url(r'^totaltrain',views.total_train),
    url(r'^coldStart', views.cold_start),
]
