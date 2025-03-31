/**
 * 图表相关的JavaScript函数
 */

// 存储图表实例以便更新
let rocChart = null;
let confusionMatrixChart = null;

/**
 * 初始化模型性能页面的图表
 * @param {number} initialModelId - 初始选中的模型ID
 */
function initializeCharts(initialModelId) {
    if (!initialModelId) return;
    
    // 初始化ROC曲线
    initializeROCChart(initialModelId);
    
    // 初始化混淆矩阵
    initializeConfusionMatrix(initialModelId);
}

/**
 * 更新所有图表
 * @param {number} modelId - 模型ID
 */
function updateCharts(modelId) {
    if (!modelId) return;
    
    // 更新ROC曲线
    updateROCChart(modelId);
    
    // 更新混淆矩阵
    updateConfusionMatrix(modelId);
}

/**
 * 初始化ROC曲线图表
 * @param {number} modelId - 模型ID
 */
function initializeROCChart(modelId) {
    const ctx = document.getElementById('roc-chart').getContext('2d');
    
    // 创建初始空图表
    rocChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'ROC曲线',
                data: [],
                borderColor: 'rgba(16, 163, 127, 1)',
                backgroundColor: 'rgba(16, 163, 127, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'ROC曲线'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `TPR: ${context.parsed.y.toFixed(3)}, FPR: ${context.parsed.x.toFixed(3)}`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '假正例率 (FPR)'
                    },
                    beginAtZero: true,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: '真正例率 (TPR)'
                    },
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // 加载初始数据
    updateROCChart(modelId);
}

/**
 * 更新ROC曲线数据
 * @param {number} modelId - 模型ID
 */
function updateROCChart(modelId) {
    fetch(`/get_roc_data/${modelId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('获取ROC数据失败:', data.error);
                return;
            }
            
            // 准备数据点
            const points = [];
            for (let i = 0; i < data.fpr.length; i++) {
                points.push({
                    x: data.fpr[i],
                    y: data.tpr[i]
                });
            }
            
            // 添加参考线的点
            points.push({x: 0, y: 0});
            points.push({x: 1, y: 1});
            
            // 排序点，确保绘图顺序正确
            points.sort((a, b) => a.x - b.x);
            
            // 更新图表数据
            rocChart.data.datasets[0].data = points;
            
            // 更新图表标题，包含AUC
            rocChart.options.plugins.title.text = `ROC曲线 (AUC = ${data.auc.toFixed(3)})`;
            
            // 更新图表
            rocChart.update();
        })
        .catch(error => {
            console.error('获取ROC数据出错:', error);
        });
}

/**
 * 初始化混淆矩阵
 * @param {number} modelId - 模型ID
 */
function initializeConfusionMatrix(modelId) {
    const ctx = document.getElementById('confusion-matrix-chart').getContext('2d');
    
    confusionMatrixChart = new Chart(ctx, {
        type: 'matrix',
        data: {
            datasets: [{
                label: '混淆矩阵',
                data: [
                    { x: '正常', y: '正常', v: 0 },
                    { x: '正常', y: '垃圾', v: 0 },
                    { x: '垃圾', y: '正常', v: 0 },
                    { x: '垃圾', y: '垃圾', v: 0 }
                ],
                backgroundColor: function(ctx) {
                    const value = ctx.dataset.data[ctx.dataIndex].v;
                    const alpha = (value === 0) ? 0.2 : 0.8;
                    return `rgba(16, 163, 127, ${alpha})`;
                },
                borderColor: 'white',
                borderWidth: 1,
                width: 20,
                height: 20
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '混淆矩阵'
                },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const data = ctx.dataset.data[ctx.dataIndex];
                            return `预测: ${data.x}, 实际: ${data.y}, 数量: ${data.v}`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '预测标签'
                    },
                    ticks: {
                        display: true
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '真实标签'
                    },
                    ticks: {
                        display: true
                    },
                    reverse: true
                }
            }
        }
    });
    
    // 加载初始数据
    updateConfusionMatrix(modelId);
}

/**
 * 更新混淆矩阵数据
 * @param {number} modelId - 模型ID
 */
function updateConfusionMatrix(modelId) {
    fetch(`/get_confusion_matrix/${modelId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('获取混淆矩阵数据失败:', data.error);
                return;
            }
            
            // 更新数据
            const matrixData = [
                { x: '正常', y: '正常', v: data.tn },
                { x: '垃圾', y: '正常', v: data.fp },
                { x: '正常', y: '垃圾', v: data.fn },
                { x: '垃圾', y: '垃圾', v: data.tp }
            ];
            
            confusionMatrixChart.data.datasets[0].data = matrixData;
            
            // 如果Chart.js没有内置matrix类型，使用替代方法
            if (!Chart.defaults.controllers.matrix) {
                displayFallbackConfusionMatrix(data);
            } else {
                confusionMatrixChart.update();
            }
            
            // 更新附加文本描述
            updateConfusionMatrixDescription(data);
        })
        .catch(error => {
            console.error('获取混淆矩阵数据出错:', error);
        });
}

/**
 * 使用HTML表格显示混淆矩阵（备用方法）
 * @param {Object} data - 混淆矩阵数据
 */
function displayFallbackConfusionMatrix(data) {
    const container = document.getElementById('confusion-matrix-fallback');
    if (!container) return;
    
    // 清空容器
    container.innerHTML = '';
    
    // 创建表格
    const table = document.createElement('table');
    table.className = 'table table-bordered confusion-matrix-table';
    
    // 表头
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th></th>
            <th colspan="2" class="text-center">预测标签</th>
        </tr>
        <tr>
            <th></th>
            <th class="text-center">正常</th>
            <th class="text-center">垃圾</th>
        </tr>
    `;
    table.appendChild(thead);
    
    // 表体
    const tbody = document.createElement('tbody');
    tbody.innerHTML = `
        <tr>
            <th rowspan="2" class="align-middle vertical-text">真实标签</th>
            <th class="text-center">正常</th>
            <td class="text-center cm-cell">${data.tn}</td>
            <td class="text-center cm-cell">${data.fp}</td>
        </tr>
        <tr>
            <th class="text-center">垃圾</th>
            <td class="text-center cm-cell">${data.fn}</td>
            <td class="text-center cm-cell">${data.tp}</td>
        </tr>
    `;
    table.appendChild(tbody);
    
    // 添加到容器
    container.appendChild(table);
    
    // 添加样式
    const style = document.createElement('style');
    style.textContent = `
        .confusion-matrix-table {
            width: auto;
            margin: 0 auto;
        }
        .cm-cell {
            font-weight: bold;
            padding: 15px 25px;
            background-color: rgba(16, 163, 127, 0.1);
        }
        .vertical-text {
            writing-mode: vertical-lr;
            transform: rotate(180deg);
            text-align: center;
        }
    `;
    document.head.appendChild(style);
    
    // 显示备用容器，隐藏canvas
    container.style.display = 'block';
    document.getElementById('confusion-matrix-chart').style.display = 'none';
}

/**
 * 更新混淆矩阵描述
 * @param {Object} data - 混淆矩阵数据
 */
function updateConfusionMatrixDescription(data) {
    const descElement = document.getElementById('confusion-matrix-description');
    if (!descElement) return;
    
    const total = data.tn + data.fp + data.fn + data.tp;
    const accuracy = ((data.tp + data.tn) / total * 100).toFixed(2);
    
    descElement.innerHTML = `
        <p>混淆矩阵显示了模型分类的详细情况：</p>
        <ul>
            <li><strong>真正例 (TP):</strong> ${data.tp} - 正确识别为垃圾短信</li>
            <li><strong>假正例 (FP):</strong> ${data.fp} - 错误地将正常短信识别为垃圾短信</li>
            <li><strong>真负例 (TN):</strong> ${data.tn} - 正确识别为正常短信</li>
            <li><strong>假负例 (FN):</strong> ${data.fn} - 错误地将垃圾短信识别为正常短信</li>
        </ul>
        <p>总分类准确率: ${accuracy}%</p>
    `;
}

// 为饼图创建一个简单的渲染函数
function renderPieChart(elementId, data, labels, colors) {
    const ctx = document.getElementById(elementId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// 创建历史记录分布图
function createHistoryDistributionChart(spamCount, normalCount) {
    return renderPieChart(
        'history-distribution-chart',
        [spamCount, normalCount],
        ['垃圾短信', '正常短信'],
        ['#F87171', '#4ADE80']
    );
}

// 加载词云图
function loadWordCloud(datasetId) {
    const wordcloudContainer = document.getElementById('wordcloud-container');
    if (!wordcloudContainer) return;
    
    // 显示加载中
    wordcloudContainer.innerHTML = '<div class="spinner"></div>';
    
    fetch(`/get_wordcloud/${datasetId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                wordcloudContainer.innerHTML = `<div class="empty-state">
                    <div class="empty-state-icon">⚠️</div>
                    <div class="empty-state-message">无法加载词云</div>
                </div>`;
                return;
            }
            
            // 设置SVG内容
            wordcloudContainer.innerHTML = data.svg;
        })
        .catch(error => {
            console.error('加载词云出错:', error);
            wordcloudContainer.innerHTML = `<div class="empty-state">
                <div class="empty-state-icon">⚠️</div>
                <div class="empty-state-message">加载词云时出错</div>
            </div>`;
        });
}
// 初始化所有图表
function initializeCharts(modelId) {
    initializeROCChart(modelId);
    initializeConfusionMatrix(modelId);
}

// 初始化ROC曲线
function initializeROCChart(modelId) {
    fetch(`/get_roc_data/${modelId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('获取ROC数据出错:', data.error);
                return;
            }

            const ctx = document.getElementById('roc-chart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.fpr.map(x => x.toFixed(2)),
                    datasets: [{
                        label: 'ROC曲线',
                        data: data.tpr,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: `ROC曲线 (AUC = ${data.auc.toFixed(3)})`
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: '假正例率 (FPR)'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: '真正例率 (TPR)'
                            }
                        }
                    }
                }
            });
        })
        .catch(error => console.error('获取ROC数据出错:', error));
}

// 初始化混淆矩阵
function initializeConfusionMatrix(modelId) {
    fetch(`/get_confusion_matrix/${modelId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('获取混淆矩阵数据出错:', data.error);
                return;
            }

            const ctx = document.getElementById('confusion-matrix-chart').getContext('2d');
            const matrix = data.matrix;
            const total = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1];

            new Chart(ctx, {
                type: 'matrix',
                data: {
                    datasets: [{
                        data: [
                            { x: 0, y: 0, v: matrix[0][0] / total },
                            { x: 1, y: 0, v: matrix[0][1] / total },
                            { x: 0, y: 1, v: matrix[1][0] / total },
                            { x: 1, y: 1, v: matrix[1][1] / total }
                        ],
                        width: ({ chart }) => (chart.chartArea || {}).width / 2 - 1,
                        height: ({ chart }) => (chart.chartArea || {}).height / 2 - 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const value = context.dataset.data[context.dataIndex];
                                    const count = matrix[value.y][value.x];
                                    const percentage = (value.v * 100).toFixed(1);
                                    return `数量: ${count} (${percentage}%)`;
                                }
                            }
                        },
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                callback: function(value) {
                                    return ['预测正常', '预测垃圾'][value];
                                }
                            }
                        },
                        y: {
                            ticks: {
                                callback: function(value) {
                                    return ['实际正常', '实际垃圾'][value];
                                }
                            }
                        }
                    }
                }
            });

            // 更新混淆矩阵描述
            const description = document.getElementById('confusion-matrix-description');
            if (description) {
                description.innerHTML = `
                    <p>混淆矩阵显示了模型的分类详情：</p>
                    <ul>
                        <li>真负例 (TN): ${matrix[0][0]} 条</li>
                        <li>假正例 (FP): ${matrix[0][1]} 条</li>
                        <li>假负例 (FN): ${matrix[1][0]} 条</li>
                        <li>真正例 (TP): ${matrix[1][1]} 条</li>
                    </ul>
                `;
            }
        })
        .catch(error => console.error('获取混淆矩阵数据出错:', error));
}
