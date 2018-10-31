import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data to a pandas dataframe.

    Args:
    messages_filepath (str)-- path to the messages file, including filename.
    categories_filepath (str) -- path to the categories file, including filename

    Returns:
    df (pandas.DataFrame) -- a dataframe consisting of messages and categories.
    """
    # Read thee files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Return a merged dataframe
    return messages.merge(categories)


def clean_data(df):
    """
    Clean the dataframe

    - Split the categories column into separate columns
    - Remove duplicates and erroneous data.

    Args:
    df (pandas.DataFrame) -- The dataframe to clean

    Returns:
    df (pandas.DataFrame) -- Clean dataframe
    """
    # Split the categories column into separate columns in a temporary dataframe
    categories = df['categories'].str.split(';', expand=True)
    # Select the first row of the categories dataframe, use this to set columns names.
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    # Clean values in all categories columns.
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # Convert column from string to numeric
        categories[column] = categories[column].astype(np.int8)

    # Drop categories column from original df, replace with categories dataframe.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Drop erroneous data and duplicates
    df = df[df.related != 2]
    df.drop_duplicates(subset='id', inplace=True, keep='last')

    return df


def save_data(df, database_filename):
    """Save df (pandas.DataFrame) to SQLite database at database_filename (str)"""
    # Create SQLAlchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # Dump data to database, replace table if it exists.
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' 
              'datasets as the first and second argument respectively, as ' 
              'well as the filepath of the database to save the cleaned data ' 
              'to as the third argument. \n\nExample: python process_data.py ' 
              'disaster_messages.csv disaster_categories.csv ' 
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
