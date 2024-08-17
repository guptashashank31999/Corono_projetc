"""event_log URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin

from covid_19 import settings
from user import views as userview

urlpatterns = [
    url(r'^admin/', admin.site.urls),

    url('^$', userview.login, name="login"),
    url('user/register', userview.register, name="register"),
    url('user/mydetails', userview.mydetails, name="mydetails"),
    url('user/updata_details', userview.updata_details, name="updata_details"),
    url('user/wellcome', userview.wellcome, name="wellcome"),
    url('user/covid', userview.covid, name="covid"),
    url('charts/(?P<chart_type>\w+)', userview.charts, name="charts"),


]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
