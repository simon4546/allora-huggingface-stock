from flask import Flask, Response, request, abort
import yfinance as yf
import torch
import json
from chronos import ChronosPipeline

app = Flask(__name__)

# 加载 chronos-t5-tiny 模型
model_name = "amazon/chronos-t5-tiny"
pipeline = ChronosPipeline.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 获取股票历史价格函数
def get_stock_history(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="5d")  # 获取最近5天的股票数据
    if not stock_info.empty:
        return stock_info['Close'].tolist()
    return None

# 生成时间序列预测
def generate_forecast(prices, prediction_length=1):
    context = torch.tensor(prices)
    forecast = pipeline.predict(context, prediction_length)
    return forecast[0].mean().item()

# 股票推理路由，股票代码通过URL参数传递
@app.route('/inference/<string:ticker>', methods=['GET'])
def inference(ticker):
    try:
        # 获取股票历史价格
        prices = get_stock_history(ticker.upper())
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
    
    # 如果价格数据不可用，返回404错误
    if prices is None:
        abort(404, description="Stock ticker not found or data unavailable")
    
    # return render_template('inference.html', ticker=ticker.upper(), prices=prices, forecast_price=forecast_price)
    try:
        # 生成未来一天的预测
        forecast_price = generate_forecast(prices, prediction_length=1)
        return Response(str(forecast_price), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
