# Required Libraries
import matplotlib.pyplot as plt

def plot_value_counts(df, col, heads_only = False):
    """Plot value counts of a column, optionally with only the heads of a household
    Parameters
	================================
	df : string
	   name of the Dataframe
    col : string
        name of the column
    returns
    =================================
    value counts plot
    """
    # Select heads of household
    if heads_only:
        df = df.loc[df['parentesco1'] == 1].copy()

    plt.figure(figsize = (8, 6))
    df[col].value_counts().sort_index().plot.bar(color = 'blue',
                                                 edgecolor = 'k',
                                                 linewidth = 2)
    plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')
    plt.show();
