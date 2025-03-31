import re
import os
import sys
import logging
from collections import Counter

# 设置环境变量以解决NumPy 2.0+的兼容性问题
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

logger = logging.getLogger(__name__)

# 尝试导入数据处理库
try:
    import jieba
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    logger.info("成功导入数据处理库")
except ImportError as e:
    logger.error(f"导入数据处理库失败: {str(e)}")
    # 这些库是必需的，所以如果导入失败就需要退出
    sys.exit(1)

# 加载停用词
def load_stopwords(file_path=None):
    """
    加载停用词列表，如果没有提供文件路径，则使用默认停用词表
    """
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                stopwords = [line.strip() for line in f]
            return set(stopwords)
        except Exception as e:
            logger.warning(f"加载停用词文件出错: {str(e)}，将使用默认停用词表")
    
    # 默认停用词表
    default_stopwords = {
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
        '或', '一个', '没有', '我们', '你们', '他们', '她们', '它们',
        '这个', '那个', '这些', '那些', '不', '在', '有', '我', '你',
        '他', '她', '它', '这', '那', '啊', '吧', '呀'
    }
    return default_stopwords

# 加载默认停用词
STOPWORDS = load_stopwords()

def preprocess_text(text, stopwords=None):
    """
    中文文本预处理：清洗、分词、去停用词
    
    参数:
    - text: 待处理的文本
    - stopwords: 停用词集合，如果为None则使用默认停用词
    
    返回:
    - 处理后的词语列表
    """
    if stopwords is None:
        stopwords = STOPWORDS
    
    if not isinstance(text, str):
        return []
    
    # 清洗文本（去除标点符号、数字、特殊字符等）
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    
    # 使用jieba进行分词
    words = jieba.cut(text)
    
    # 去除停用词
    words = [w for w in words if w.strip() and w not in stopwords and len(w) > 1]
    
    return words

def extract_features(texts, max_features=5000):
    """
    使用TF-IDF提取文本特征
    
    参数:
    - texts: 文本列表
    - max_features: 特征词汇最大数量
    
    返回:
    - 特征矩阵和TF-IDF向量化器
    """
    # 预处理文本
    processed_texts = []
    for text in texts:
        words = preprocess_text(text)
        processed_texts.append(' '.join(words))
    
    # TF-IDF特征提取
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(processed_texts)
    
    return features, vectorizer

def preprocess_dataset(df):
    """
    预处理数据集，包括文本处理和特征提取
    
    参数:
    - df: 包含'text'和'label'列的DataFrame
    
    返回:
    - 预处理后的特征和标签，以及TF-IDF向量化器
    """
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # 提取特征
    features, vectorizer = extract_features(texts)
    
    return features, np.array(labels), vectorizer

def create_word_cloud(texts, output_path=None, max_words=200, width=800, height=500):
    """
    从文本列表创建词云图
    
    参数:
    - texts: 文本列表
    - output_path: 输出文件路径，如果为None则返回WordCloud对象
    - max_words: 最大词数
    - width: 词云图宽度
    - height: 词云图高度
    
    返回:
    - 如果output_path为None，返回WordCloud对象，否则返回None
    """
    # 预处理所有文本
    all_words = []
    for text in texts:
        words = preprocess_text(text)
        all_words.extend(words)
    
    # 统计词频
    word_counts = Counter(all_words)
    
    # 获取系统字体
    font_path = get_system_font()
    
    try:
        # 如果没有找到合适的字体，直接使用SVG生成方法
        if font_path is None:
            logger.warning("找不到合适的字体，使用SVG方式生成词云")
            if output_path:
                create_simple_svg_wordcloud(word_counts, max_words, output_path)
            return None
        
        # 创建词云
        logger.info(f"使用字体: {font_path} 创建词云")
        wordcloud = WordCloud(
            font_path=font_path,
            width=width, 
            height=height,
            max_words=max_words,
            background_color='white',
            # 增加一些参数以提高中文显示效果
            prefer_horizontal=0.9,
            max_font_size=150,
            min_font_size=10,
            collocations=False,
            random_state=42
        ).generate_from_frequencies(word_counts)
        
        if output_path:
            plt.figure(figsize=(width/100, height/100), dpi=300)  # 提高DPI
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_path, format='svg', bbox_inches='tight')
            plt.close()
            return None
        else:
            return wordcloud
    except Exception as e:
        logger.error(f"创建词云时出错: {str(e)}")
        # 创建一个简单的替代词云SVG
        if output_path:
            create_simple_svg_wordcloud(word_counts, max_words, output_path)
        return None

def create_simple_svg_wordcloud(word_counts, max_words, output_path):
    """创建简单的SVG词云"""
    # 选择频率最高的n个词
    top_words = dict(word_counts.most_common(max_words))
    if not top_words:
        logger.warning("词频统计为空，无法创建词云")
        # 创建一个带有提示信息的空白SVG
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="500" xmlns="http://www.w3.org/2000/svg">
<rect width="800" height="500" fill="white"/>
<text x="400" y="250" font-size="24" text-anchor="middle" fill="gray">暂无足够数据生成词云</text>
</svg>'''
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg)
        return
    
    max_count = max(top_words.values())
    
    # SVG头部
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="500" xmlns="http://www.w3.org/2000/svg">
<rect width="800" height="500" fill="white"/>
<style>
    text {{
        font-family: "DejaVu Sans", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei", sans-serif;
    }}
</style>
'''
    
    # 随机放置词语
    import random
    
    # 使用更多的颜色
    colors = [
        "#4285F4", "#EA4335", "#FBBC05", "#34A853",  # Google colors
        "#3498DB", "#E74C3C", "#2ECC71", "#F39C12",  # Flat UI colors
        "#9B59B6", "#1ABC9C", "#E67E22", "#16A085",  # More colors
        "#2980B9", "#D35400", "#27AE60", "#8E44AD",  # Even more colors
    ]
    
    # 创建网格布局
    grid_size = int(max_words ** 0.5) + 1  # 计算网格大小
    cell_width = 800 // grid_size
    cell_height = 500 // grid_size
    
    # 对词按频率排序
    words_sorted = sorted(top_words.items(), key=lambda x: x[1], reverse=True)
    
    for i, (word, count) in enumerate(words_sorted):
        if i >= max_words:
            break
            
        # 计算位置
        row = i // grid_size
        col = i % grid_size
        
        # 加入随机偏移使词云更自然
        x_offset = random.randint(-10, 10)
        y_offset = random.randint(-10, 10)
        
        x = col * cell_width + cell_width // 2 + x_offset
        y = row * cell_height + cell_height // 2 + y_offset
        
        # 根据词频确定字体大小
        size = 10 + (count / max_count) * 50
        
        # 颜色随机选择
        color = random.choice(colors)
        
        # 随机旋转（但不要旋转太多，以保持可读性）
        rotate = random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, -15]) 
        
        transform = f'transform="rotate({rotate} {x} {y})"' if rotate != 0 else ''
        
        # 确保显示在中心
        svg += f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="middle" dominant-baseline="middle" {transform}>{word}</text>\n'
    
    # SVG尾部
    svg += '</svg>'
    
    # 保存SVG文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg)

import os
# 检查系统中文字体
def get_system_font():
    """检查系统中文字体"""
    # 首先检查我们是否有复制的字体
    if os.path.exists('static/fonts/DejaVuSans.ttf'):
        logger.info("使用复制的DejaVuSans字体")
        return 'static/fonts/DejaVuSans.ttf'
    
    # 检查系统字体
    potential_fonts = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux DejaVu
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Noto CJK
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
        '/System/Library/Fonts/PingFang.ttc',  # macOS
        'C:/Windows/Fonts/simhei.ttf',  # Windows
        'C:/Windows/Fonts/msyh.ttf',  # Windows
        '/home/runner/workspace/.pythonlibs/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf'  # 默认matplotlib字体
    ]
    
    for font_path in potential_fonts:
        if os.path.exists(font_path):
            logger.info(f"使用系统字体: {font_path}")
            return font_path
    
    logger.warning("找不到合适的字体，将使用默认字体")
    return None
