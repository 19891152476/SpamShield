import datetime
import json
from app import db

class Dataset(db.Model):
    """数据集模型"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    file_path = db.Column(db.String(255), nullable=False)
    rows_count = db.Column(db.Integer, default=0)
    spam_count = db.Column(db.Integer, default=0)
    normal_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # 关系
    models = db.relationship('Model', backref='dataset', lazy=True)
    
    def __repr__(self):
        return f'<Dataset {self.name}>'

class Model(db.Model):
    """模型记录"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # "naive_bayes", "svm", "lstm", "residual_lstm"
    file_path = db.Column(db.String(255), nullable=False)
    metrics = db.Column(db.Text)  # 存储为JSON字符串
    train_time = db.Column(db.Float)  # 训练时间（秒）
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    is_active = db.Column(db.Boolean, default=False)  # 是否为当前激活模型
    
    # 外键
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    
    # 关系
    detections = db.relationship('DetectionHistory', backref='model', lazy=True)
    
    def __repr__(self):
        return f'<Model {self.name} ({self.model_type})>'
    
    def get_metrics(self):
        """将存储的JSON字符串转换为字典"""
        if self.metrics:
            return json.loads(self.metrics)
        return {}
    
    def set_metrics(self, metrics_dict):
        """将指标字典转换为JSON字符串存储"""
        self.metrics = json.dumps(metrics_dict)

class DetectionHistory(db.Model):
    """检测历史记录"""
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    is_spam = db.Column(db.Boolean, nullable=False)  # 检测结果: True=垃圾, False=正常
    confidence = db.Column(db.Float)  # 置信度 (0-1)
    detected_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    source = db.Column(db.String(50))  # 'single' 或 'batch'
    
    # 外键
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'), nullable=False)
    
    def __repr__(self):
        return f'<Detection {"Spam" if self.is_spam else "Normal"} {self.detected_at}>'

class BatchDetection(db.Model):
    """批量检测记录"""
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255), nullable=False)
    total_messages = db.Column(db.Integer, default=0)
    spam_count = db.Column(db.Integer, default=0)
    normal_count = db.Column(db.Integer, default=0)
    result_path = db.Column(db.String(255))  # 结果文件保存路径
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f'<BatchDetection {self.file_name} ({self.total_messages} messages)>'
