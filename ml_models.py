import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from preprocessing import preprocess_dataset, preprocess_text
import logging

logger = logging.getLogger(__name__)

# 标记TensorFlow是否可用
tensorflow_available = False

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout, Bidirectional, Add, Multiply, Activation, Concatenate
    tensorflow_available = True
    logger.info("TensorFlow 导入成功")
except (ImportError, TypeError) as e:
    logger.warning(f"TensorFlow 导入失败: {str(e)}")
    logger.warning("深度学习模型将不可用，但SVM和朴素贝叶斯模型依然可用")

def train_model(dataset_path, model_type, output_path, params=None):
    """
    训练指定类型的模型
    
    参数:
    - dataset_path: 数据集文件路径
    - model_type: 模型类型 ('naive_bayes', 'svm', 'lstm', 'residual_lstm')
    - output_path: 模型输出路径
    - params: 模型参数字典
    
    返回:
    - 保存的模型路径和性能指标字典
    """
    # 默认参数
    if params is None:
        params = {}
    
    # 设置默认值
    epochs = params.get('epochs', 5)
    batch_size = params.get('batch_size', 32)
    learning_rate = params.get('learning_rate', 0.001)
    test_split = params.get('test_split', 0.2)
    
    # 加载数据集
    df = pd.read_csv(dataset_path)
    
    # 确保必要的列存在
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("数据集缺少必要的'text'或'label'列")
    
    # 预处理数据
    X, y, vectorizer = preprocess_dataset(df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    
    # 创建模型
    model = None
    model_data = {}
    metrics = {}
    
    if model_type == 'naive_bayes':
        # 朴素贝叶斯模型
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'model_type': 'naive_bayes'
        }
        
    elif model_type == 'svm':
        # SVM模型
        model = SVC(probability=True)
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'model_type': 'svm'
        }
        
    elif model_type in ['lstm', 'residual_lstm']:
        if not tensorflow_available:
            raise ValueError(f"TensorFlow不可用，无法训练{model_type}模型。请尝试使用 'naive_bayes' 或 'svm' 模型类型。")
            
        # 重新处理原始文本用于深度学习模型
        texts = df['text'].tolist()
        processed_texts = [' '.join(preprocess_text(text)) for text in texts]
        
        # 使用Tokenizer准备数据
        max_words = 10000  # 词汇表大小
        max_len = 100  # 序列最大长度
        
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(processed_texts)
        
        sequences = tokenizer.texts_to_sequences(processed_texts)
        X_padded = pad_sequences(sequences, maxlen=max_len)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=test_split, random_state=42)
        
        # 创建深度学习模型
        if model_type == 'lstm':
            # 标准LSTM模型
            model = Sequential([
                Embedding(max_words, 128, input_length=max_len),
                Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
        else:  # residual_lstm
            # 创建带残差连接和注意力机制的LSTM模型
            inputs = Input(shape=(max_len,))
            embedding = Embedding(max_words, 128, input_length=max_len)(inputs)
            
            # 双向LSTM层
            lstm_out = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(embedding)
            
            # 注意力机制
            attention = Dense(1, activation='tanh')(lstm_out)
            attention = Activation('softmax')(attention)
            attention = Multiply()([lstm_out, attention])
            attention_sum = tf.keras.backend.sum(attention, axis=1)
            
            # 残差连接
            lstm_out2 = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(lstm_out)
            concat = Concatenate()([attention_sum, lstm_out2])
            
            # 全连接层
            x = Dense(64, activation='relu')(concat)
            x = Dropout(0.3)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 评估模型
        y_proba = model.predict(X_test).flatten()
        y_pred = (y_proba > 0.5).astype(int)
        
        # 保存模型额外信息
        model_data = {
            'model': model,
            'tokenizer': tokenizer,
            'max_len': max_len,
            'model_type': model_type,
            'history': history.history
        }
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 计算通用性能指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # 整理性能指标
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(roc_auc),
        'fpr': [float(x) for x in fpr],
        'tpr': [float(x) for x in tpr],
        'confusion_matrix': cm
    }
    
    # 如果是深度学习模型，添加训练历史
    if model_type in ['lstm', 'residual_lstm'] and 'history' in model_data:
        metrics['train_loss'] = model_data['history']['loss']
        metrics['val_loss'] = model_data['history']['val_loss']
        metrics['train_acc'] = model_data['history']['accuracy']
        metrics['val_acc'] = model_data['history']['val_accuracy']
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存模型
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"模型 {model_type} 训练完成，保存至 {output_path}")
    logger.info(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
    
    return output_path, metrics

def load_model_from_file(model_path):
    """
    从文件加载模型
    
    参数:
    - model_path: 模型文件路径
    
    返回:
    - 加载的模型数据
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        raise

def predict_message(message, model_obj=None, model_path=None):
    """
    预测单条短信是否为垃圾短信
    
    参数:
    - message: 短信内容
    - model_obj: 模型对象
    - model_path: 模型文件路径
    
    返回:
    - (预测结果, 置信度) - 结果为布尔值，True表示垃圾短信
    """
    if model_obj is None and model_path is None:
        raise ValueError("必须提供model_obj或model_path之一")
    
    if model_obj is not None:
        # 使用数据库模型记录
        model_path = model_obj.file_path
    
    # 加载模型
    model_data = load_model_from_file(model_path)
    model_type = model_data['model_type']
    model = model_data['model']
    
    # 根据模型类型进行预测
    if model_type in ['naive_bayes', 'svm']:
        # 传统机器学习模型
        vectorizer = model_data['vectorizer']
        
        # 预处理文本
        words = preprocess_text(message)
        processed_text = ' '.join(words)
        
        # 转换为特征向量
        X = vectorizer.transform([processed_text])
        
        # 预测
        y_pred = model.predict(X)[0]
        y_proba = model.predict_proba(X)[0, 1]  # 获取垃圾短信的概率
        
        return bool(y_pred), float(y_proba)
    
    elif model_type in ['lstm', 'residual_lstm']:
        if not tensorflow_available:
            raise ValueError(f"TensorFlow不可用，无法使用{model_type}模型进行预测。请使用朴素贝叶斯或支持向量机模型。")
            
        # 深度学习模型
        tokenizer = model_data['tokenizer']
        max_len = model_data['max_len']
        
        # 预处理文本
        words = preprocess_text(message)
        processed_text = ' '.join(words)
        
        # 转换为序列
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len)
        
        # 预测
        y_proba = float(model.predict(padded_sequence)[0, 0])
        y_pred = y_proba > 0.5
        
        return bool(y_pred), y_proba
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def predict_batch(df, model_obj):
    """
    批量预测短信
    
    参数:
    - df: 包含'text'列的DataFrame
    - model_obj: 模型对象
    
    返回:
    - 包含预测结果的DataFrame
    """
    if 'text' not in df.columns:
        raise ValueError("DataFrame必须包含'text'列")
    
    # 复制输入DataFrame以避免修改原始数据
    result_df = df.copy()
    
    # 添加预测结果列
    result_df['is_spam'] = False
    result_df['confidence'] = 0.0
    
    # 逐条预测
    for idx, row in result_df.iterrows():
        try:
            is_spam, confidence = predict_message(row['text'], model_obj)
            result_df.at[idx, 'is_spam'] = is_spam
            result_df.at[idx, 'confidence'] = confidence
        except Exception as e:
            logger.error(f"预测第{idx}行时出错: {str(e)}")
            result_df.at[idx, 'is_spam'] = False
            result_df.at[idx, 'confidence'] = 0.0
    
    return result_df
