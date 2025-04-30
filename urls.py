from django.urls import path, re_path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Your API routes
    path('api/', include('yourapp.urls')),
    # Serve React app
    re_path(r'^.*', TemplateView.as_view(template_name='index.html'), name='home'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)