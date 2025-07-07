import os

print("=== 服务开始启动 ===")
print("当前目录内容：", os.listdir("."))
print("OER_model.joblib 存在吗？", os.path.exists("OER_model.joblib"))
print("ORR_model.joblib 存在吗？", os.path.exists("ORR_model.joblib"))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import copy

app = Flask(__name__)
CORS(app)


app = Flask(__name__)
CORS(app)

# 加载模型
oer_model = joblib.load('OER_model.joblib')
orr_model = joblib.load('ORR_model.joblib')

# 特征名必须与训练时完全一致
feature_names = [
    '体系的能量', 'M的形成能', 'M的结合能', 'M4的形成能', 'M4的结合能', 'M溶解电势', 'M4溶解电势',
    'M的内聚能', 'M4的内聚能', '结合前单原子内聚能', '结合前团簇内聚能', 'OH在单原子上的Bader电荷',
    'OOH在单原子上的Bader电荷', 'OH在团簇上的Bader电荷', 'OOH在团簇上的Bader电荷',
    '单原子的d带中心', '团簇的d带中心', 'M的bader电荷', '结合前M的bader电荷', 'M4的bader电荷',
    '结合前M4的bader电荷', 'OH', 'O', 'OOH', 'O-OH', 'OOH-O'
]
@app.route('/predict.html')
def serve_predict_html():
    return send_from_directory('.', 'predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # 取特征值，保证顺序
        features = [float(data[name]) for name in feature_names]
        X = np.array(features).reshape(1, -1)
        oer_pred = oer_model.predict(X)[0]
        orr_pred = orr_model.predict(X)[0]

        # ---------- 自动建议：敏感性分析 ----------
        step = 0.05  # 5%微调
        oer_delta = {}
        orr_delta = {}
        for i, fname in enumerate(feature_names):
            # OER
            new_features_up = copy.deepcopy(features)
            new_features_down = copy.deepcopy(features)
            new_features_up[i] *= (1 + step)
            new_features_down[i] *= (1 - step)
            oer_up = oer_model.predict(np.array(new_features_up).reshape(1, -1))[0]
            oer_down = oer_model.predict(np.array(new_features_down).reshape(1, -1))[0]
            if oer_up < oer_pred:
                oer_delta[fname + "↑"] = oer_pred - oer_up
            if oer_down < oer_pred:
                oer_delta[fname + "↓"] = oer_pred - oer_down
            # ORR
            orr_up = orr_model.predict(np.array(new_features_up).reshape(1, -1))[0]
            orr_down = orr_model.predict(np.array(new_features_down).reshape(1, -1))[0]
            if orr_up < orr_pred:
                orr_delta[fname + "↑"] = orr_pred - orr_up
            if orr_down < orr_pred:
                orr_delta[fname + "↓"] = orr_pred - orr_down

        advice = []
        if oer_pred > 1.5:
            oer_best = sorted(oer_delta.items(), key=lambda x: -x[1])[:3]
            if oer_best:
                best_str = "，".join([f"{k}（可降{v:.3f}）" for k, v in oer_best])
                advice.append(f"OER过电势偏高，建议优先调整：{best_str}")
        if orr_pred > 1.0:
            orr_best = sorted(orr_delta.items(), key=lambda x: -x[1])[:3]
            if orr_best:
                best_str = "，".join([f"{k}（可降{v:.3f}）" for k, v in orr_best])
                advice.append(f"ORR过电势偏高，建议优先调整：{best_str}")
        if not advice:
            advice.append("性能良好，无需优化。")
        # ----------------------------------------

        return jsonify({'OER': float(oer_pred), 'ORR': float(orr_pred), '建议': advice})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # 如果环境变量 PORT 存在，则用它，否则用 5000
    app.run(host="0.0.0.0", port=port)
