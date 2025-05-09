#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os


class Config:
    """
    ���ù�����
    """

    def __init__(self, config_path="config.json"):
        # Ĭ������
        self.default_config = {
            "server": {
                "host": "localhost",
                "port": 8765
            },
            "llm": {
                "model_name": "deepseek-ai/deepseek-llm-7b-chat",
                "model_path": "./models/deepseek-llm-7b-chat",
                "use_lora": True,
                "lora_path": "./models/lora"
            },
            "emotion": {
                "audio_model_path": "./models/audio_emotion",
                "face_model_path": "./models/face_emotion",
                "fusion_weights": {
                    "audio": 0.4,
                    "face": 0.6
                }
            },
            "knowledge": {
                "neo4j": {
                    "uri": "bolt://localhost:7687",
                    "user": "neo4j",
                    "password": "password"
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

        # ���������ļ�
        self.config_path = config_path
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self._merge_configs(self.default_config, user_config)
        else:
            # ����Ĭ������
            self.save_config()

        self.config = self.default_config

    def _merge_configs(self, default, user):
        """
        �ϲ�����
        """
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value

    def save_config(self):
        """
        �������õ��ļ�
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        """
        ��ȡ����
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
