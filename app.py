import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 定义SQLAlchemy基类
class Base(DeclarativeBase):
    pass

# 初始化SQLAlchemy
db = SQLAlchemy(model_class=Base)

# 创建Flask应用
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "spam_filter_secret_key")

# 配置数据库
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# 初始化应用
db.init_app(app)

# 确保应用上下文中导入模型，以便创建数据表
with app.app_context():
    # 导入模型
    import models  
    
    # 创建所有表
    db.create_all()
    logger.info("数据库表创建完成")

# 导入路由
from routes import *

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
