import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import logging

logger = logging.getLogger(__name__)

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_filename(base_name, extension, prefix='', suffix=''):
    """生成带时间戳的文件名"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{base_name}_{timestamp}{suffix}.{extension}" if prefix else f"{base_name}_{timestamp}{suffix}.{extension}"
    return filename

def plot_to_base64(fig):
    """将matplotlib图形转换为base64字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def plot_confusion_matrix(cm, class_names=['正常', '垃圾']):
    """
    绘制混淆矩阵
    
    参数:
    - cm: 混淆矩阵数组
    - class_names: 类别名称
    
    返回:
    - base64编码的图像字符串
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 显示混淆矩阵
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置轴标签
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='真实标签',
        xlabel='预测标签'
    )
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 显示数值
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # 转换为base64
    img_str = plot_to_base64(fig)
    plt.close(fig)
    
    return img_str

def plot_roc_curve(fpr, tpr, auc_value):
    """
    绘制ROC曲线
    
    参数:
    - fpr: 假正例率数组
    - tpr: 真正例率数组
    - auc_value: AUC值
    
    返回:
    - base64编码的图像字符串
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 绘制ROC曲线
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_value:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正例率 (FPR)')
    ax.set_ylabel('真正例率 (TPR)')
    ax.set_title('接收者操作特征 (ROC) 曲线')
    ax.legend(loc="lower right")
    
    # 转换为base64
    img_str = plot_to_base64(fig)
    plt.close(fig)
    
    return img_str

def plot_training_history(history):
    """
    绘制训练历史曲线
    
    参数:
    - history: 训练历史字典
    
    返回:
    - base64编码的图像字符串
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制损失曲线
    ax1.plot(history['loss'], label='训练损失')
    ax1.plot(history['val_loss'], label='验证损失')
    ax1.set_title('模型损失')
    ax1.set_ylabel('损失')
    ax1.set_xlabel('轮次')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(history['accuracy'], label='训练准确率')
    ax2.plot(history['val_accuracy'], label='验证准确率')
    ax2.set_title('模型准确率')
    ax2.set_ylabel('准确率')
    ax2.set_xlabel('轮次')
    ax2.legend()
    
    # 转换为base64
    img_str = plot_to_base64(fig)
    plt.close(fig)
    
    return img_str

def format_time(seconds):
    """格式化时间（秒）为人类可读格式"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"
