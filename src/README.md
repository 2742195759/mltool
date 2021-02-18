# 机器学习分析软件

## 接口
python 接口:
文件处理方法

python3 loadcsv.py --input ./output.csv  # done
python3 predict.py --input ./output.csv --output ./tmp.csv --ylabel 'label_name' --xlabels ['x1', 'x2', 'x3'] --ml-method 'classifier'
python3 to_enmu.py --input ./output.csv --output ./tmp.csv --dict '{"xk":1, "xg":2}' --column 'age'
python3 normalization.py --input ./output.csv --output ./tmp.csv --column 'age' --method "sdf"
