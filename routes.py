import os
import datetime
import pandas as pd
import numpy as np
import json
from flask import render_template, request, redirect, url_for, flash, jsonify, session, send_file
from werkzeug.utils import secure_filename
from app import app, db
from models import Dataset, Model, DetectionHistory, BatchDetection
from preprocessing import preprocess_text, create_word_cloud
from ml_models import train_model, predict_message, predict_batch, load_model_from_file
import logging
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import base64

# 配置日志
logger = logging.getLogger(__name__)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'saved_models'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'csv', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 首页 - 短信检测界面
@app.route('/')
def index():
    # 检查是否有激活的模型
    active_model = Model.query.filter_by(is_active=True).first()
    models = Model.query.all()
    return render_template('index.html', active_model=active_model, models=models)

# 单条短信检测
@app.route('/detect', methods=['POST'])
def detect():
    message = request.form.get('message', '')
    if not message:
        flash('请输入短信内容', 'error')
        return redirect(url_for('index'))
    
    # 获取当前激活的模型
    active_model = Model.query.filter_by(is_active=True).first()
    if not active_model:
        flash('请先训练或加载模型', 'error')
        return redirect(url_for('index'))
    
    try:
        # 使用模型进行预测
        result, confidence = predict_message(message, active_model)
        
        # 保存检测记录
        detection = DetectionHistory(
            message=message,
            is_spam=result,
            confidence=confidence,
            model_id=active_model.id,
            source='single'
        )
        db.session.add(detection)
        db.session.commit()
        
        result_text = "垃圾短信" if result else "正常短信"
        flash(f'检测结果: {result_text} (置信度: {confidence:.2f})', 'success' if not result else 'warning')
    except Exception as e:
        logger.error(f"检测错误: {str(e)}")
        flash(f'检测过程中发生错误: {str(e)}', 'error')
    
    return redirect(url_for('index'))

# 批量检测
@app.route('/detect_batch', methods=['POST'])
def detect_batch():
    # 检查是否有激活的模型
    active_model = Model.query.filter_by(is_active=True).first()
    if not active_model:
        flash('请先训练或加载模型', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('未找到文件', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('未选择文件', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 读取文件并进行批量检测
            df = pd.read_csv(file_path)
            if 'text' not in df.columns:
                flash('文件格式错误: 必须包含"text"列', 'error')
                return redirect(url_for('index'))
            
            # 进行批量检测
            results_df = predict_batch(df, active_model)
            
            # 生成结果文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"result_{timestamp}_{filename}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            
            # 保存结果文件
            results_df.to_csv(result_path, index=False)
            
            # 统计垃圾短信和正常短信数量
            spam_count = results_df[results_df['is_spam'] == True].shape[0]
            normal_count = results_df[results_df['is_spam'] == False].shape[0]
            
            # 保存批量检测记录
            batch = BatchDetection(
                file_name=filename,
                total_messages=len(results_df),
                spam_count=spam_count,
                normal_count=normal_count,
                result_path=result_path
            )
            db.session.add(batch)
            
            # 添加每条短信的检测历史
            for _, row in results_df.iterrows():
                detection = DetectionHistory(
                    message=row['text'],
                    is_spam=row['is_spam'],
                    confidence=row['confidence'],
                    model_id=active_model.id,
                    source='batch'
                )
                db.session.add(detection)
            
            db.session.commit()
            
            flash(f'批量检测完成，共检测 {len(results_df)} 条短信，其中垃圾短信 {spam_count} 条，正常短信 {normal_count} 条', 'success')
            
            # 返回下载链接
            return render_template(
                'index.html', 
                active_model=active_model, 
                models=Model.query.all(),
                batch_result=True,
                result_path=result_filename,
                total_messages=len(results_df),
                spam_count=spam_count,
                normal_count=normal_count
            )
            
        except Exception as e:
            logger.error(f"批量检测错误: {str(e)}")
            flash(f'批量检测过程中发生错误: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('不支持的文件类型，请上传CSV文件', 'error')
        return redirect(url_for('index'))

# 下载批量检测结果
@app.route('/download_result/<filename>')
def download_result(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

# 数据集管理页面
@app.route('/dataset')
def dataset():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template('dataset.html', datasets=datasets)

# 上传数据集
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        flash('未找到文件', 'error')
        return redirect(url_for('dataset'))
    
    file = request.files['file']
    if file.filename == '':
        flash('未选择文件', 'error')
        return redirect(url_for('dataset'))
    
    if file and allowed_file(file.filename):
        try:
            # 保存文件
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # 读取数据集
            df = pd.read_csv(file_path)
            
            # 检查必要的列
            if 'text' not in df.columns or 'label' not in df.columns:
                flash('文件格式错误: 必须包含"text"和"label"列', 'error')
                return redirect(url_for('dataset'))
            
            # 获取数据集统计信息
            total_rows = len(df)
            spam_count = df[df['label'] == 1].shape[0]
            normal_count = df[df['label'] == 0].shape[0]
            
            # 创建数据集记录
            dataset_name = request.form.get('name', filename)
            dataset_desc = request.form.get('description', '')
            
            new_dataset = Dataset(
                name=dataset_name,
                description=dataset_desc,
                file_path=file_path,
                rows_count=total_rows,
                spam_count=spam_count,
                normal_count=normal_count
            )
            
            db.session.add(new_dataset)
            db.session.commit()
            
            # 生成词云（异步生成会更好，但为简单起见，这里同步生成）
            spam_messages = df[df['label'] == 1]['text'].tolist()
            if spam_messages:
                try:
                    wordcloud_path = os.path.join(app.config['UPLOAD_FOLDER'], f"wordcloud_{new_dataset.id}.svg")
                    create_word_cloud(spam_messages, wordcloud_path)
                except Exception as e:
                    logger.error(f"生成词云错误: {str(e)}")
            
            flash(f'数据集 "{dataset_name}" 上传成功, 共 {total_rows} 条短信，其中垃圾短信 {spam_count} 条，正常短信 {normal_count} 条', 'success')
        except Exception as e:
            logger.error(f"上传数据集错误: {str(e)}")
            flash(f'上传数据集过程中发生错误: {str(e)}', 'error')
    else:
        flash('不支持的文件类型，请上传CSV文件', 'error')
    
    return redirect(url_for('dataset'))

# 删除数据集
@app.route('/delete_dataset/<int:dataset_id>', methods=['POST'])
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # 检查是否有关联的模型
    if dataset.models:
        flash('无法删除此数据集，因为它关联了已训练的模型', 'error')
        return redirect(url_for('dataset'))
    
    try:
        # 删除实际文件
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # 删除词云文件（如果存在）
        wordcloud_path = os.path.join(app.config['UPLOAD_FOLDER'], f"wordcloud_{dataset.id}.svg")
        if os.path.exists(wordcloud_path):
            os.remove(wordcloud_path)
        
        # 删除数据库记录
        db.session.delete(dataset)
        db.session.commit()
        
        flash(f'数据集 "{dataset.name}" 已成功删除', 'success')
    except Exception as e:
        logger.error(f"删除数据集错误: {str(e)}")
        flash(f'删除数据集过程中发生错误: {str(e)}', 'error')
    
    return redirect(url_for('dataset'))

# 模型训练
@app.route('/train_model', methods=['POST'])
def train_model_route():
    dataset_id = request.form.get('dataset_id')
    model_type = request.form.get('model_type')
    model_name = request.form.get('model_name')
    
    # 验证输入
    if not all([dataset_id, model_type, model_name]):
        flash('请填写所有必要的字段', 'error')
        return redirect(url_for('model'))
    
    # 获取训练参数
    params = {
        'epochs': int(request.form.get('epochs', 5)),
        'batch_size': int(request.form.get('batch_size', 32)),
        'learning_rate': float(request.form.get('learning_rate', 0.001)),
        'test_split': float(request.form.get('test_split', 0.2)),
    }
    
    # 获取数据集
    dataset = Dataset.query.get_or_404(dataset_id)
    
    try:
        # 训练模型
        start_time = datetime.datetime.now()
        model_path, metrics = train_model(
            dataset.file_path, 
            model_type, 
            os.path.join(app.config['MODEL_FOLDER'], f"{model_name}_{model_type}.pkl"),
            params
        )
        end_time = datetime.datetime.now()
        train_duration = (end_time - start_time).total_seconds()
        
        # 创建模型记录
        new_model = Model(
            name=model_name,
            model_type=model_type,
            file_path=model_path,
            metrics=json.dumps(metrics),
            train_time=train_duration,
            dataset_id=dataset.id
        )
        
        # 如果这是第一个模型，自动激活它
        if Model.query.count() == 0:
            new_model.is_active = True
        
        db.session.add(new_model)
        db.session.commit()
        
        flash(f'模型 "{model_name}" 训练成功，准确率: {metrics["accuracy"]:.2f}', 'success')
    except Exception as e:
        logger.error(f"训练模型错误: {str(e)}")
        flash(f'训练模型过程中发生错误: {str(e)}', 'error')
    
    return redirect(url_for('model'))

# 模型管理页面
@app.route('/model')
def model():
    models = Model.query.order_by(Model.created_at.desc()).all()
    datasets = Dataset.query.all()
    return render_template('model.html', models=models, datasets=datasets)

# 激活模型
@app.route('/activate_model/<int:model_id>', methods=['POST'])
def activate_model(model_id):
    # 取消所有模型的激活状态
    Model.query.update({Model.is_active: False})
    
    # 激活选定的模型
    model = Model.query.get_or_404(model_id)
    model.is_active = True
    
    db.session.commit()
    
    flash(f'模型 "{model.name}" 已激活', 'success')
    return redirect(url_for('model'))

# 删除模型
@app.route('/delete_model/<int:model_id>', methods=['POST'])
def delete_model(model_id):
    model = Model.query.get_or_404(model_id)
    
    try:
        # 检查是否有关联的检测历史
        if DetectionHistory.query.filter_by(model_id=model_id).count() > 0:
            flash('无法删除此模型，因为它关联了历史检测记录', 'error')
            return redirect(url_for('model'))
        
        # 删除实际文件
        if os.path.exists(model.file_path):
            os.remove(model.file_path)
        
        # 删除数据库记录
        db.session.delete(model)
        db.session.commit()
        
        flash(f'模型 "{model.name}" 已成功删除', 'success')
    except Exception as e:
        logger.error(f"删除模型错误: {str(e)}")
        flash(f'删除模型过程中发生错误: {str(e)}', 'error')
    
    return redirect(url_for('model'))

# 历史记录页面
@app.route('/history')
def history():
    # 获取搜索参数
    search = request.args.get('search', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    # 基础查询
    query = DetectionHistory.query
    
    # 应用搜索过滤
    if search:
        query = query.filter(DetectionHistory.message.like(f'%{search}%'))
    
    # 应用日期过滤
    if start_date:
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        query = query.filter(DetectionHistory.detected_at >= start_datetime)
    
    if end_date:
        end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        end_datetime = end_datetime + datetime.timedelta(days=1)  # 包含当天
        query = query.filter(DetectionHistory.detected_at < end_datetime)
    
    # 排序和分页
    page = request.args.get('page', 1, type=int)
    per_page = 20
    pagination = query.order_by(DetectionHistory.detected_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    # 获取统计数据
    total_count = DetectionHistory.query.count()
    spam_count = DetectionHistory.query.filter_by(is_spam=True).count()
    normal_count = DetectionHistory.query.filter_by(is_spam=False).count()
    
    # 生成饼图数据
    spam_percentage = (spam_count / total_count * 100) if total_count > 0 else 0
    normal_percentage = (normal_count / total_count * 100) if total_count > 0 else 0
    
    return render_template(
        'history.html', 
        pagination=pagination,
        search=search,
        start_date=start_date,
        end_date=end_date,
        total_count=total_count,
        spam_count=spam_count,
        normal_count=normal_count,
        spam_percentage=spam_percentage,
        normal_percentage=normal_percentage
    )

# 模型性能页面
@app.route('/performance')
def performance():
    models = Model.query.all()
    selected_model_id = request.args.get('model_id', type=int)
    
    if selected_model_id:
        selected_model = Model.query.get_or_404(selected_model_id)
    elif models:
        selected_model = models[0]
    else:
        selected_model = None
    
    return render_template(
        'performance.html', 
        models=models,
        selected_model=selected_model
    )

# 获取模型ROC曲线数据
@app.route('/get_roc_data/<int:model_id>')
def get_roc_data(model_id):
    model = Model.query.get_or_404(model_id)
    metrics = model.get_metrics()
    
    if 'fpr' in metrics and 'tpr' in metrics:
        return jsonify({
            'fpr': metrics['fpr'],
            'tpr': metrics['tpr'],
            'auc': metrics.get('auc', 0)
        })
    else:
        return jsonify({
            'error': '没有可用的ROC曲线数据'
        }), 404

# 获取模型混淆矩阵数据
@app.route('/get_confusion_matrix/<int:model_id>')
def get_confusion_matrix(model_id):
    model = Model.query.get_or_404(model_id)
    metrics = model.get_metrics()
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        return jsonify({
            'matrix': cm,
            'tn': cm[0][0],
            'fp': cm[0][1],
            'fn': cm[1][0],
            'tp': cm[1][1]
        })
    else:
        return jsonify({
            'error': '没有可用的混淆矩阵数据'
        }), 404

# API端点: 获取模型性能指标
@app.route('/api/model_metrics/<int:model_id>')
def api_model_metrics(model_id):
    model = Model.query.get_or_404(model_id)
    metrics = model.get_metrics()
    
    basic_metrics = {
        'accuracy': metrics.get('accuracy', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
    }
    
    return jsonify(basic_metrics)

# 生成词云图API
@app.route('/get_wordcloud/<int:dataset_id>')
def get_wordcloud(dataset_id):
    wordcloud_path = os.path.join(app.config['UPLOAD_FOLDER'], f"wordcloud_{dataset_id}.svg")
    
    if os.path.exists(wordcloud_path):
        with open(wordcloud_path, 'r') as f:
            svg_content = f.read()
        return jsonify({'svg': svg_content})
    else:
        return jsonify({'error': '词云图不存在'}), 404

# 404 错误处理
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error='页面未找到'), 404

# 500 错误处理
@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='服务器内部错误'), 500
