import click
import pandas as pd
from dataprep.eda import create_report


@click.command()
@click.option('--path_to_data',
              default='data/raw/heart_cleveland_upload.csv',
              type=click.Path(exists=True))
@click.option('--path_to_save',
              default='report/eda.html',
              type=click.Path())
def generate_eda_report(path_to_data, path_to_save):
    df = pd.read_csv(path_to_data)
    report = create_report(df)
    report.save(path_to_save)


if __name__ == '__main__':
    generate_eda_report()
