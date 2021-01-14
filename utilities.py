from IPython.display import display_html
def display_side_by_side(*args):
        html_str=''
        for df in args:
                html_str+=df.to_html()
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)   

def kaggle(api):
    # é preciso ter o arquivo kaggle.json na mesma pasta
    os.system('cmd /k "mkdir -p ~/.kaggle"')
    os.system('cmd /k "cp kaggle.json ~/.kaggle/"')
    os.system('cmd /k "chmod 600 ~/.kaggle/kaggle.json"')

def copy_columns(X, variables, copies=1, copy_labels=None):
  if copy_labels:
    for label in copy_labels:
      for variable in variables:
        X[f'{label}_{variable}'] = X[variable]