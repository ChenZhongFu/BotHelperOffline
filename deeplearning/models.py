# -*- coding:utf-8 -*-
from __future__ import unicode_literals

from django.db import models

class ModelInfo(models.Model):
    class Meta:
        db_table = 'model_info'
    id = models.BigAutoField(max_length=20, primary_key=True)
    online_url = models.CharField(max_length=200)
    offline_url = models.CharField(max_length=200)
    is_online = models.SmallIntegerField()    # 0:离线,1:在线
    is_training = models.SmallIntegerField()  # 0:没在训练,1:正在训练
    app_id = models.CharField(max_length=200)
    is_replace = models.SmallIntegerField()

class ModelHistory(models.Model):
    class Meta:
        db_table = 'model_history'
    id = models.BigAutoField(primary_key=True)
    model_id = models.BigIntegerField()
    train_data_num = models.IntegerField()
    accuracy = models.FloatField()
    loss = models.FloatField()
    train_start_time = models.DateTimeField()
    train_cost_time = models.TimeField()
