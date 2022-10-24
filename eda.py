from ipywidgets import Dropdown,interact
import matplotlib.pyplot as plt
import seaborn as sns
def make_plots(df):
    data = df.select_dtypes(exclude='object')
    print('SELECT FEATURE TO EXPLORE BELOW: ')
    def make_nominal_plot(feature):
        num_feat = df[[feature,'target']]
        fig,ax = plt.subplots(ncols=2,figsize=(25,9))
        sns.boxplot(data=num_feat,x =feature,y='target',ax=ax[0])
        sns.histplot(num_feat[feature],ax=ax[1])
        box_string = f'A BoxPlot for Distribution of {num_feat.columns[0]} for each Target Class'
        hist_string = f'A Histogram for Distribution of {num_feat.columns[0]} Feature'
        ax[0].set_title(box_string)
        ax[1].set_title(hist_string)
        plt.tight_layout()
    thresh_widget = Dropdown(options=data.columns,value = df.columns[0])
    interact(make_nominal_plot,feature=thresh_widget)
        
        