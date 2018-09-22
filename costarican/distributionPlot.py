# Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Function for distribution plot
def kde_target(df, variable):
    """Plots the distribution of `variable` in `df` colored by the `Target` column"""

    colors = {1: 'red', 2: 'orange', 3: 'blue', 4: 'green'}

    plt.figure(figsize = (12, 8))

    df = df[df['Target'].notnull()]

    for level in df['Target'].unique():
        subset = df[df['Target'] == level].copy()
        sns.kdeplot(subset[variable].dropna(),
                    label = f'Poverty Level: {level}',
                    color = colors[int(subset['Target'].unique())])

    plt.xlabel(variable); plt.ylabel('Density');
    plt.title('{} Distribution'.format(variable.capitalize()));
