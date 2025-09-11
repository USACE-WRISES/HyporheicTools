import sys, ast
paths = [
    'src/HyporheicTools/inputs.py',
    'src/HyporheicTools/functions/my_utils.py',
    'src/HyporheicTools/core/run_from_yaml.py',
    'src/HyporheicTools/esri/toolboxes/Hyporheic_Toolbox.pyt',
]
for p in paths:
    try:
        with open(p, 'r', encoding='utf-8') as f:
            src = f.read()
        ast.parse(src, filename=p)
        print('OK', p)
    except SyntaxError as e:
        print('SyntaxError', p, e)
        raise
