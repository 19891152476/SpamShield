/**
 * 垃圾短信过滤系统前端脚本
 */

// DOM 加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 初始化提示框
    initializeTooltips();
    
    // 初始化表单验证
    initializeFormValidation();
    
    // 文件上传预览
    initializeFileUpload();
    
    // 初始化首次使用引导（如果需要）
    checkFirstTimeUser();
    
    // 初始时隐藏加载动画
    const spinners = document.querySelectorAll('.spinner');
    spinners.forEach(spinner => {
        spinner.style.display = 'none';
    });
    
    // 处理警告消息自动消失
    initializeAlerts();
    
    // 初始化动态元素
    initializeDynamicElements();
});

/**
 * 初始化提示框
 */
function initializeTooltips() {
    const tooltips = document.querySelectorAll('[data-toggle="tooltip"]');
    tooltips.forEach(tooltip => {
        // 这里简单实现，实际项目可以使用更复杂的提示框库
        const tip = document.createElement('span');
        tip.className = 'tooltip-text';
        tip.textContent = tooltip.getAttribute('title');
        tooltip.classList.add('tooltip');
        tooltip.appendChild(tip);
    });
}

/**
 * 初始化表单验证
 */
function initializeFormValidation() {
    const forms = document.querySelectorAll('form[data-validate="true"]');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const requiredFields = form.querySelectorAll('[required]');
            let isValid = true;
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('is-invalid');
                    
                    // 如果字段旁边没有错误消息，添加一个
                    let errorMsg = field.nextElementSibling;
                    if (!errorMsg || !errorMsg.classList.contains('error-message')) {
                        errorMsg = document.createElement('div');
                        errorMsg.className = 'error-message text-danger';
                        errorMsg.textContent = '此字段是必填的';
                        field.parentNode.insertBefore(errorMsg, field.nextSibling);
                    }
                } else {
                    field.classList.remove('is-invalid');
                    
                    // 移除错误消息
                    const errorMsg = field.nextElementSibling;
                    if (errorMsg && errorMsg.classList.contains('error-message')) {
                        errorMsg.remove();
                    }
                }
            });
            
            if (!isValid) {
                event.preventDefault();
            } else {
                // 显示加载动画
                const spinner = form.querySelector('.spinner');
                if (spinner) {
                    spinner.style.display = 'block';
                }
                
                // 禁用提交按钮
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.disabled = true;
                }
            }
        });
        
        // 实时验证
        const requiredFields = form.querySelectorAll('[required]');
        requiredFields.forEach(field => {
            field.addEventListener('input', function() {
                if (field.value.trim()) {
                    field.classList.remove('is-invalid');
                    
                    // 移除错误消息
                    const errorMsg = field.nextElementSibling;
                    if (errorMsg && errorMsg.classList.contains('error-message')) {
                        errorMsg.remove();
                    }
                }
            });
        });
    });
}

/**
 * 初始化文件上传预览
 */
function initializeFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const fileLabel = document.querySelector(`label[for="${input.id}"]`);
            if (fileLabel) {
                if (input.files.length > 0) {
                    fileLabel.textContent = input.files[0].name;
                } else {
                    fileLabel.textContent = '选择文件';
                }
            }
            
            // 显示文件名
            const fileNameDisplay = document.getElementById(`${input.id}-name`);
            if (fileNameDisplay) {
                if (input.files.length > 0) {
                    fileNameDisplay.textContent = `已选择: ${input.files[0].name}`;
                    fileNameDisplay.style.display = 'block';
                } else {
                    fileNameDisplay.style.display = 'none';
                }
            }
        });
    });
}

/**
 * 检查是否首次使用系统
 */
function checkFirstTimeUser() {
    // 检查本地存储中是否有首次使用标记
    const isFirstTime = localStorage.getItem('firstTimeUser') === null;
    
    if (isFirstTime) {
        // 如果是首页并且没有模型，显示引导
        const noModelWarning = document.querySelector('.no-model-warning');
        if (noModelWarning) {
            showFirstTimeGuide();
            // 设置标记，下次不再显示
            localStorage.setItem('firstTimeUser', 'false');
        }
    }
}

/**
 * 显示首次使用引导
 */
function showFirstTimeGuide() {
    // 创建引导覆盖层
    const guideOverlay = document.createElement('div');
    guideOverlay.className = 'guide-overlay';
    
    // 创建引导模态框
    const guideModal = document.createElement('div');
    guideModal.className = 'guide-modal';
    guideModal.innerHTML = `
        <h3 class="guide-title">欢迎使用垃圾短信过滤系统</h3>
        <p>看起来这是您首次使用本系统。请按照以下步骤开始使用：</p>
        
        <div class="guide-steps">
            <div class="guide-step">
                <div class="step-number">1</div>
                <div class="step-content">
                    <div class="step-title">上传数据集</div>
                    <div class="step-desc">首先，前往"数据集管理"页面，上传包含短信内容和标签的CSV文件。</div>
                </div>
            </div>
            
            <div class="guide-step">
                <div class="step-number">2</div>
                <div class="step-content">
                    <div class="step-title">训练模型</div>
                    <div class="step-desc">在"模型管理"页面，选择一个数据集和算法类型来训练模型。</div>
                </div>
            </div>
            
            <div class="guide-step">
                <div class="step-number">3</div>
                <div class="step-content">
                    <div class="step-title">开始检测</div>
                    <div class="step-desc">回到首页，使用已训练的模型进行单条或批量短信检测。</div>
                </div>
            </div>
        </div>
        
        <button class="btn btn-primary btn-block" id="guide-close-btn">我知道了</button>
    `;
    
    guideOverlay.appendChild(guideModal);
    document.body.appendChild(guideOverlay);
    
    // 关闭按钮事件
    document.getElementById('guide-close-btn').addEventListener('click', function() {
        guideOverlay.remove();
    });
}

/**
 * 初始化警告消息自动消失
 */
function initializeAlerts() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        // 添加关闭按钮
        if (!alert.querySelector('.close-btn')) {
            const closeBtn = document.createElement('button');
            closeBtn.className = 'close-btn';
            closeBtn.innerHTML = '&times;';
            closeBtn.style.float = 'right';
            closeBtn.style.background = 'none';
            closeBtn.style.border = 'none';
            closeBtn.style.cursor = 'pointer';
            closeBtn.style.fontSize = '1.25rem';
            closeBtn.addEventListener('click', function() {
                alert.remove();
            });
            alert.insertBefore(closeBtn, alert.firstChild);
        }
        
        // 5秒后自动消失
        setTimeout(() => {
            alert.style.opacity = '0';
            alert.style.transition = 'opacity 0.5s';
            setTimeout(() => {
                alert.remove();
            }, 500);
        }, 5000);
    });
}

/**
 * 初始化动态元素
 */
function initializeDynamicElements() {
    // 初始化模型选择交互
    const modelSelector = document.getElementById('model-selector');
    if (modelSelector) {
        modelSelector.addEventListener('change', function() {
            const selectedOption = this.options[this.selectedIndex];
            const modelId = selectedOption.value;
            const modelType = selectedOption.getAttribute('data-type');
            
            // 更新模型类型显示
            const modelTypeDisplay = document.getElementById('selected-model-type');
            if (modelTypeDisplay) {
                modelTypeDisplay.textContent = getModelTypeDisplayName(modelType);
            }
            
            // 更新性能指标（如果在性能页面）
            if (window.location.pathname.includes('/performance')) {
                fetchModelMetrics(modelId);
            }
        });
    }
    
    // 初始化数据集选择交互
    const datasetSelector = document.getElementById('dataset-selector');
    if (datasetSelector) {
        datasetSelector.addEventListener('change', function() {
            // 可以在这里添加额外的交互逻辑
        });
    }
    
    // 初始化模型类型选择交互
    const modelTypeSelector = document.getElementById('model-type');
    if (modelTypeSelector) {
        modelTypeSelector.addEventListener('change', function() {
            const selectedType = this.value;
            
            // 显示或隐藏深度学习特定参数
            const dlParams = document.getElementById('dl-params');
            if (dlParams) {
                if (selectedType === 'lstm' || selectedType === 'residual_lstm') {
                    dlParams.style.display = 'block';
                } else {
                    dlParams.style.display = 'none';
                }
            }
            
            // 更新模型说明
            updateModelDescription(selectedType);
        });
        
        // 初始触发一次
        if (modelTypeSelector.value) {
            updateModelDescription(modelTypeSelector.value);
        }
    }
}

/**
 * 获取模型类型的显示名称
 */
function getModelTypeDisplayName(modelType) {
    const typeMap = {
        'naive_bayes': '朴素贝叶斯',
        'svm': '支持向量机',
        'lstm': '长短期记忆网络',
        'residual_lstm': '残差注意力LSTM'
    };
    
    return typeMap[modelType] || modelType;
}

/**
 * 更新模型说明
 */
function updateModelDescription(modelType) {
    const descriptionElement = document.getElementById('model-description');
    if (!descriptionElement) return;
    
    let description = '';
    
    switch (modelType) {
        case 'naive_bayes':
            description = '朴素贝叶斯是一种基于贝叶斯定理的分类算法，适用于文本分类任务，训练速度快，性能较好。';
            break;
        case 'svm':
            description = '支持向量机是一种强大的分类算法，通过寻找最佳分隔超平面来区分不同类别，适用于高维特征空间。';
            break;
        case 'lstm':
            description = '长短期记忆网络是一种特殊的循环神经网络，能够捕捉序列数据中的长期依赖关系，适用于文本等序列数据。';
            break;
        case 'residual_lstm':
            description = '残差注意力LSTM结合了残差连接和注意力机制，能够更好地捕捉中文短信中的上下文信息和重要特征，分类精度更高。';
            break;
        default:
            description = '请选择一个模型类型。';
    }
    
    descriptionElement.textContent = description;
}

/**
 * 异步加载模型的性能指标
 */
function fetchModelMetrics(modelId) {
    if (!modelId) return;
    
    fetch(`/api/model_metrics/${modelId}`)
        .then(response => response.json())
        .then(data => {
            // 更新指标显示
            document.getElementById('accuracy-value').textContent = (data.accuracy * 100).toFixed(2) + '%';
            document.getElementById('precision-value').textContent = (data.precision * 100).toFixed(2) + '%';
            document.getElementById('recall-value').textContent = (data.recall * 100).toFixed(2) + '%';
            document.getElementById('f1-value').textContent = (data.f1 * 100).toFixed(2) + '%';
            
            // 触发图表更新
            if (typeof updateCharts === 'function') {
                updateCharts(modelId);
            }
        })
        .catch(error => {
            console.error('获取模型指标失败:', error);
        });
}

/**
 * 确认删除
 */
function confirmDelete(message, formId) {
    if (confirm(message || '确认要删除吗？此操作不可撤销。')) {
        document.getElementById(formId).submit();
    }
    return false;
}

/**
 * 激活模型
 */
function activateModel(formId) {
    document.getElementById(formId).submit();
}
