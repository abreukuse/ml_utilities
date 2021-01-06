from IPython.display import display_html
def display_side_by_side(*args):
        html_str=''
        for df in args:
                html_str+=df.to_html()
        display_html(html_str.replace('table','table style="display:inline"'),raw=True)
        
def agrupar(dataframe, agrupamento):
    return pd.concat({grupo: resultado for grupo, resultado in dataframe.groupby(agrupamento)})

def kaggle(api):
    # é preciso ter o arquivo kaggle.json na mesma pasta
    os.system('cmd /k "mkdir -p ~/.kaggle"')
    os.system('cmd /k "cp kaggle.json ~/.kaggle/"')
    os.system('cmd /k "chmod 600 ~/.kaggle/kaggle.json"')